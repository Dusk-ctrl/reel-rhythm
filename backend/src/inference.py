"""Inference helper. The real Gemma model is NOT loaded inside Netlify."""
from pathlib import Path

MODEL_ID = "google/gemma-4-E4B-it"
ADAPTERS = {"sft": "outputs/sft/final_adapter", "rl": "outputs/rl/final_adapter"}


class GemmaGenerator:
    def __init__(self, variant="base", adapter_path=None):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma GPU inference is not configured; use the verified rule demo instead.")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto"
        )
        if variant != "base":
            adapter = adapter_path or ADAPTERS.get(variant)
            if not adapter or not Path(adapter).exists():
                raise FileNotFoundError(f"Adapter for {variant} is missing. Train it first.")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, prompt, seed=1, sample=True):
        import torch
        # Gemma 4 uses a multimodal processor, even for text-only prompts.
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        start = inputs["input_ids"].shape[-1]
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=128,
                do_sample=sample, temperature=.8 if sample else None,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self.processor.decode(out[0][start:], skip_special_tokens=True)
