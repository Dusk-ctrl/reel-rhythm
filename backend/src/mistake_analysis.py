"""Collect real changed answers; prefer early-vs-late RL checkpoints when available.

The assignment asks for at least three early mistakes that are corrected later.
If those cases are not present in the runs, I report the limitation honestly.
"""
import json
from pathlib import Path


def main():
    root = Path("outputs/evaluations")
    checkpoints = sorted(root.glob("rl_test_step*.json"),
                         key=lambda path: int(path.stem.split("step")[-1]))
    if len(checkpoints) >= 2:
        early_path, late_path = checkpoints[0], checkpoints[-1]
        comparison = "early RL checkpoint -> late RL checkpoint"
    else:
        early_path, late_path = root / "base_test.json", root / "rl_test.json"
        comparison = "base model -> final RL (not equivalent to early-vs-late RL checkpoints)"
    if not (early_path.exists() and late_path.exists()):
        raise FileNotFoundError("First evaluate base/RL or two RL checkpoints on the same test set.")
    early = json.loads(early_path.read_text(encoding="utf-8"))
    late = json.loads(late_path.read_text(encoding="utf-8"))
    old = {row["id"]: row for row in early["results"]}
    examples = []
    for after in late["results"]:
        before = old.get(after["id"])
        if before and not before["pass1"] and after["pass1"]:
            examples.append({
                "id": after["id"], "prompt": after["prompt"],
                "expected": after["expected"],
                "before_response": before["attempts"][0]["response"],
                "after_response": after["attempts"][0]["response"],
            })
    output = {"comparison": comparison, "cases": examples, "count": len(examples)}
    (root / "corrected_mistakes.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Comparison: {comparison}")
    print(f"Found {len(examples)} real corrected cases. The assignment requires at least 3.")
    if len(examples) < 3:
        print("Do not claim 3 corrected mistakes; document the limitation or perform further runs.")


if __name__ == "__main__":
    main()
