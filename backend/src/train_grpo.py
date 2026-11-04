"""GRPO experiment: Gemma generates attempts and learns from the scores.

The ground-truth letter stays hidden from the model prompt. The reward function
can see it to grade each sampled completion. This is the required Option A loop.
"""
import json
from pathlib import Path

from src.data_prep import read_jsonl
from src.hardware import require_training_gpu
from src.reward import grpo_reward

MODEL_ID = "google/gemma-4-E4B-it"


def train_grpo(max_steps=40):
    require_training_gpu()
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = read_jsonl("data/processed/train.jsonl")
    if not rows:
        raise RuntimeError("Please prepare the real datasets before training.")
    # This dataset does NOT give reference solutions to the model.
    samples = Dataset.from_list([
        {"prompt": row["prompt"], "ground_truth": row["ground_truth"]} for row in rows
    ])
    args = GRPOConfig(
        output_dir="outputs/rl",
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        max_completion_length=128,
        num_generations=4,
        beta=.02,  # Small KL penalty limits drift from the reference policy.
        remove_unused_columns=False,  # Keep ground_truth for the reward checker.
        logging_steps=5,
        save_steps=max(10, max_steps // 4),
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16"},
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    print("GRPO from BASE Gemma (not from the SFT adapter). Expect 4 answers per prompt.")
    trainer = GRPOTrainer(
        model=MODEL_ID, args=args, train_dataset=samples,
        reward_funcs=grpo_reward, peft_config=lora,
    )
    trainer.train()
    trainer.save_model("outputs/rl/final_adapter")
    Path("outputs/rl").mkdir(parents=True, exist_ok=True)
    (Path("outputs/rl") / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8"
    )
    print("RL adapter saved: outputs/rl/final_adapter")


def main():
    train_grpo()


if __name__ == "__main__":
    main()
