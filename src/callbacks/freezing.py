from transformers import TrainerCallback

class MaskCallback(TrainerCallback):

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        total_elems = 0
        zero_elems = 0

        for name, param in model.named_parameters():
            # Use safe_get_full_grad instead of param.grad
            grad = param.grad
            
            if grad is None:
                print("No grad for ", name)
                continue

            total_elems += grad.numel()
            zero_elems += (grad == 0).sum().item()

        pct = 100 * zero_elems / total_elems if total_elems > 0 else 0.0
        print(f"[GradZeroCheck] Zero grads: {zero_elems}/{total_elems} ({pct:.4f}%)")

