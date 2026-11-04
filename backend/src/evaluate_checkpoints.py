"""Measure accuracy and answer length over saved training checkpoints.

Use validation for development; reserve test until the experiment is frozen.
"""
import argparse
import json
import re
from pathlib import Path

from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sft", "rl"], default="rl")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    paths = sorted((Path("outputs") / args.model).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not paths:
        raise FileNotFoundError("No checkpoints saved yet. Run training before plotting checkpoint accuracy.")
    summaries = []
    for path in paths:
        step = int(path.name.split("-")[-1])
        print("Evaluating checkpoint at step", step)
        row = evaluate_model(args.model, k=args.k, split=args.split,
                             adapter_path=str(path), output_suffix=f"_step{step}")
        row["step"] = step
        summaries.append(row)
    destination = Path("outputs/evaluations") / f"{args.model}_{args.split}_checkpoints.json"
    destination.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("Saved checkpoint curve data:", destination)


if __name__ == "__main__":
    main()
