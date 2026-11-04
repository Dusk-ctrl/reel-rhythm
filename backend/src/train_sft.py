"""SFT baseline: learn from correct solutions made by our deterministic checker.

I train on the same TRAIN questions that GRPO uses, not validation/test.
SFT copies reference completions, whereas RL learns from attempt rewards.
"""
from pathlib import Path

from src.data_prep import read_jsonl
from src.hardware import require_training_gpu

MODEL_ID = "google/gemma-4-E4B-it"


def train_sft(max_steps=40):
    require_training_gpu()  # Avoid accidental CPU training.
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = read_jsonl("data/processed/train.jsonl")
    if not rows:
        raise RuntimeError("No training questions found; first run --mode prepare")
    # A reference is computed by the exact same Python rule as ground_truth.
    examples = [{"prompt": row["prompt"], "completion": row["reference"]} for row in rows]
    dataset = Dataset.from_list(examples)
    args = SFTConfig(
        output_dir="outputs/sft",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        bf16=True,
        max_length=512,
        completion_only_loss=True,
        logging_steps=5,
        save_steps=max(10, max_steps // 4),
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16"},
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    print("Starting SFT on training questions only. This may require version-specific Gemma 4 tuning.")
    trainer = SFTTrainer(model=MODEL_ID, args=args, train_dataset=dataset, peft_config=lora)
    trainer.train()
    trainer.save_model("outputs/sft/final_adapter")
    Path("outputs/sft").mkdir(parents=True, exist_ok=True)
    import json
    (Path("outputs/sft") / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8"
    )
    print("SFT adapter saved: outputs/sft/final_adapter")


def main():
    train_sft()


if __name__ == "__main__":
    main()
