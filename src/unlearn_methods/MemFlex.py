from unlearn_methods import UnlearningMethod, register_method
import torch
import json
import numpy as np
from pretrain.localization import compute_cosine_similarity, compute_info
from pretrain.data_module import convert_to_model_format_with_random_label, custom_data_collator
from torch.utils.data import Dataset
import os
from pretrain.config import add_dataset_index
from llm_unlearn.utils import tokenize, AdvSupervisedDataset
from llm_unlearn.methods.ascent_plus_descent import AscentPlusDescentTrainer, AscentPlusDescentDataCollator
from dataclasses import dataclass, field, fields
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments

@dataclass
class DataArguments:
    positive_ratio: int = field(
        default=1, 
        metadata={"help": "Number of positive examples per negative example."}
    )
    positive_factor: float = field(
        default=1.0, 
        metadata={"help": "The loss weight factor for positive examples."}
    )


class TextDatasetRandomQA(Dataset):
    def __init__(self, ds, tokenizer, model_cfg, max_length=512, question_key='text', answer_key='labels',):
        super(TextDatasetRandomQA, self).__init__()
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.max_length = max_length
        self.data = ds
        self.data = add_dataset_index(self.data)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_to_model_format_with_random_label(self.tokenizer, self.max_length, question, answer, self.model_cfg)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(), \
                torch.stack(label_list).squeeze(), \
                torch.stack(pad_attention_mask_list).squeeze(), \
                torch.tensor(indices)



@register_method("memflex")
class MemFlex(UnlearningMethod):
    
    def format_dataset(self, dataset):

        tpl_args = self.config.model.template_args
        
        u_start = tpl_args.user_start_tag   
        u_end   = tpl_args.user_end_tag     
        a_start = tpl_args.asst_start_tag   
        a_end   = tpl_args.asst_end_tag   

        def apply_template(example):

            user_part = f"{u_start}{example['question']}{u_end}"
            
            asst_part = f"{a_start}{example['answer']}{a_end}"
            
            return {
                "text": f"{user_part}{asst_part}",
                "labels": asst_part
            }
        return dataset.map(apply_template)
    
    def _get_save_dir(self, data):
        field = data.cfg.forget.Forget.name
        method_name = "memflex"
        return os.path.join(self.config.paths.output_dir, "unlearned_models", field, method_name)

    def tokenize_data(self, tokenizer, data):
        tokenized_data_path = os.path.join(self._get_save_dir(data), "tokenized_dataset.pt")
        
        model_max_length = self.config.model.get("model_max_length", 512)
        
        positive_data = self.format_dataset(data.retain)
        negative_data = self.format_dataset(data.forget)
        
        negative_dataset = tokenize(negative_data, tokenizer, model_max_length)
        positive_dataset = tokenize(positive_data, tokenizer, model_max_length)
        
        valid_keys = {f.name for f in fields(DataArguments)}
        filtered_args = {k: v for k, v in self.config.items() if k in valid_keys}
        data_args = DataArguments(**filtered_args)
        train_dataset = AdvSupervisedDataset(negative_dataset, positive_dataset, data_args)
        torch.save(train_dataset, tokenized_data_path)
        return

    def localization(self, model, tokenizer, data):
        save_dir = self._get_save_dir(data)
        os.makedirs(save_dir, exist_ok=True)

        for split in ["retention", "unlearn"]:
            if os.path.exists(os.path.join(save_dir, f"grad_info_{split}.pt")):
                print(f"Grad info for {split} already exists, skipping computation.")
                continue
            if split == "retention":
                split_data = data.retain
            elif split == "unlearn":
                split_data = data.forget

            tpl = self.config.model.template_args
            model_cfg = {
                'question_start_tag': tpl.user_start_tag,
                'question_end_tag': tpl.user_end_tag,
                'answer_tag': tpl.asst_start_tag,
            }
            torch_format_dataset = TextDatasetRandomQA(split_data, tokenizer=tokenizer, model_cfg=model_cfg, max_length=512, question_key='question', answer_key='answer')

            num_devices = int(os.environ.get('WORLD_SIZE', 1))
            print(f"num_devices: {num_devices}")

            # load dataloader
            train_dataloader = torch.utils.data.DataLoader(
            torch_format_dataset,
            batch_size=self.config.batch_size,
            collate_fn=custom_data_collator,
            shuffle=True,
            num_workers=4,
        )
            # calculate info matrix
            info_matrix = compute_info(model, train_dataloader)

            # save info_matrix
            torch.save(info_matrix, os.path.join(save_dir, f"grad_info_{split}.pt"))


        grad_retention = torch.load(os.path.join(save_dir, "grad_info_retention.pt"))
        grad_unlearn = torch.load(os.path.join(save_dir, "grad_info_unlearn.pt"))
        
        # Localization
        delta_matrix = {}

        unlearn_list = []
        retention_list = []
        item_list = []
        for k, _ in grad_unlearn.items():
            if k in grad_retention:
                delta_matrix[k] = compute_cosine_similarity(grad_unlearn[k], grad_retention[k]).squeeze()
                num_unlearn = np.mean(np.abs(grad_unlearn[k].numpy()))
                num_retention = np.mean(np.abs(grad_retention[k].numpy()))
                unlearn_list.append(num_unlearn)
                retention_list.append(num_retention)
                item_list.append(delta_matrix[k])

        sim_thre = self.config.sim_thresh
        grad_thre = self.config.grad_thresh
        item_array = np.array(item_list)
        unlearn_array = np.array(unlearn_list)
        unlearn_sim_idx = np.where(item_array < sim_thre)[0]
        unlearn_grad_idx = np.where(unlearn_array > grad_thre)[0]

        located_region_num = list(np.intersect1d(unlearn_sim_idx, unlearn_grad_idx))
        located_region = []
        for i, key in enumerate(grad_unlearn.keys()):
            if i in located_region_num:
                located_region.append(key)

        print(f"[MemFlex] Localized {len(located_region)} / {len(grad_unlearn)} LoRA params "
              f"(sim_thresh={sim_thre}, grad_thresh={grad_thre})")
        print(f"[MemFlex]   grad magnitudes: min={unlearn_array.min():.2e}, "
              f"max={unlearn_array.max():.2e}, median={np.median(unlearn_array):.2e}")
        print(f"[MemFlex]   cosine sims:    min={item_array.min():.3f}, "
              f"max={item_array.max():.3f}, median={np.median(item_array):.3f}")
        if len(located_region) == 0:
            raise RuntimeError(
                f"MemFlex localized 0 params — training would be a no-op. "
                f"Lower grad_thresh (currently {grad_thre}; max observed grad is "
                f"{unlearn_array.max():.2e}) or raise sim_thresh (currently {sim_thre}; "
                f"min observed cosine sim is {item_array.min():.3f})."
            )

        with open(os.path.join(save_dir, f"located_region.json"), "w") as f:
            json.dump(located_region, f, indent=4)
        return

    def _get_peft_model(self, model):
        """Add LoRA adapters to the model (or load existing ones)."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(self.config.lora_target_modules),
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model.enable_input_require_grads()
        lora_adapter_path = os.path.join(self._get_save_dir_from_config(), "lora_adapter")
        if os.path.exists(lora_adapter_path):
            model = PeftModel.from_pretrained(model, lora_adapter_path)
        else:
            model = get_peft_model(model, peft_config)
        return model

    def _get_save_dir_from_config(self):
        return self.config.paths.output_dir

    def apply_memflex(self, model, tokenizer, data):
        save_dir = self._get_save_dir(data)

        # LOAD LOCATED WEIGHTS FOR UPDATE AND MAKE OTHER WEIGHTS FROZEN
        with open(os.path.join(save_dir, f"located_region.json"), "r") as f:
            unlearn_region = json.load(f)
        for n, p in model.named_parameters():
            if n in unlearn_region:
                p.requires_grad = True
            else:
                p.requires_grad = False
                

        #TOKENIZE DATA IF NOT PREPROCESSED BEFORE
        tokenized_data_path = os.path.join(save_dir, "tokenized_dataset.pt")
        if not os.path.exists(tokenized_data_path):
            self.tokenize_data(tokenizer, data)
              
        train_dataset = torch.load(tokenized_data_path, weights_only=False)
        if self.config.data_args.get("max_train_samples", None) is not None:
            max_train_samples = min(len(train_dataset), self.config.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
         
        training_args = TrainingArguments(**self.config.training_args)
        training_args.unlearn_method = self.config.method
        training_args.memflex_forget_factor = self.config.get("forget_factor", -0.4)
        training_args.memflex_retain_factor = self.config.get("retain_factor", 2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        # define Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get("cosine_t_max", 150),
        )
        # pass optimizer and scheduler into trainer
        Trainer_args = {
            "args": training_args,
        }
        Trainer_args["args"].optimizers = (optimizer, scheduler)
        unlearner = AscentPlusDescentTrainer(
            model=model,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            **Trainer_args,
            data_collator=AscentPlusDescentDataCollator(tokenizer),
        )

        # Add EvalDashboardCallback if configured
        eval_dashboard_cfg = self.config.get("eval_dashboard", {})
        if eval_dashboard_cfg.get("enabled", False):
            from callbacks.EvalDashboardCallback import EvalDashboardCallback
            lm_eval_cfg = eval_dashboard_cfg.get("lm_eval", {})
            panorama_cfg = eval_dashboard_cfg.get("panorama", {})

            panorama_eval_cfg = None
            if hasattr(self, 'full_cfg') and self.full_cfg is not None:
                eval_cfgs = self.full_cfg.get("eval", {})
                panorama_eval_cfg = eval_cfgs.get("panorama", None)

            template_args = self.config.model.template_args
            callback = EvalDashboardCallback(
                lm_eval_tasks=list(lm_eval_cfg.get("tasks", [])),
                lm_eval_interval=lm_eval_cfg.get("interval", 50),
                lm_eval_batch_size=lm_eval_cfg.get("batch_size", 8),
                lm_eval_limit=lm_eval_cfg.get("limit", None),
                panorama_eval_cfg=panorama_eval_cfg,
                panorama_interval=panorama_cfg.get("interval", 50),
                tokenizer=tokenizer,
                template_args=template_args,
                eval_on_start=eval_dashboard_cfg.get("eval_on_start", True),
                eval_on_end=eval_dashboard_cfg.get("eval_on_end", True),
                save_dir=getattr(self, 'full_cfg', {}).get("paths", {}).get("output_dir", ".") if hasattr(self, 'full_cfg') else ".",
            )
            unlearner.add_callback(callback)

        unlearner.train()
        model = unlearner.model
        model.save_pretrained(os.path.join(save_dir, f"unlearned_lora_adapter"))
        merged_model = model.merge_and_unload()
        return merged_model
    
    def unlearn(self, model, tokenizer, data):
        

        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.do_sample = True
        
        print("-"*50)
        print(self.config.paths)
        print("-"*50)
        
        
        model = self._get_peft_model(model)

        self.localization(model, tokenizer, data)
        return self.apply_memflex(model, tokenizer, data)