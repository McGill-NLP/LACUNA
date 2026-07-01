from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(model_cfg):
    

    model_name = model_cfg.get("name", None)
    model_path = model_cfg.get("path", None)
    torch_dtype = model_cfg.get("torch_dtype", None)
    revision = model_cfg.get("revision", None)
    attn_implementation = model_cfg.get("attn_implementation", None)

    if model_name is None and model_path is None:
        raise ValueError("Either 'name' or 'path' must be specified in model config.")

    load_kwargs = {}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if revision is not None:
        load_kwargs["revision"] = revision
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    if model_path is None:
        model_path = model_name
        
        
    print(f"Loading model from {model_path} with args: {load_kwargs}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)        

    return model, tokenizer