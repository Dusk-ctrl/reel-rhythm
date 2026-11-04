"""Evaluate base/SFT/RL with exactly the same untouched test questions.

pass@1 = one sampled answer is right. pass@4 = any of four is right.
I save mistakes to compare earlier and later models honestly.
"""
import json
import statistics
from pathlib import Path

from src.data_prep import read_jsonl
from src.reward import grade_response


def evaluate_model(variant="base", k=4, split="test", adapter_path=None, output_suffix=""):
    if split not in ("validation", "test"):
        raise ValueError("Only validation and test are used for evaluation")
    rows = read_jsonl(f"data/processed/{split}.jsonl")
    from src.inference import GemmaGenerator
    generator = GemmaGenerator(variant, adapter_path=adapter_path)
    results = []
    for n, row in enumerate(rows):
        attempts = []
        for trial in range(k):
            response = generator.generate(row["prompt"], seed=1000 + n * 31 + trial)
            scored = grade_response(response, row["ground_truth"])
            attempts.append({"response": response, **scored})
        selected = attempts[0]["prediction"]
        selected_title = row["candidate_titles"]["ABCD".index(selected)] if selected and selected in "ABCD" else None
        results.append({
            "selected_title": selected_title,
            "id": row["id"], "category": row["category"],
            "expected": row["ground_truth"], "title": row["expected_title"],
            "prompt": row["prompt"], "attempts": attempts,
            "pass1": int(attempts[0]["correct"]),
            "passk": int(any(a["correct"] for a in attempts)),
        })
        if (n + 1) % 10 == 0:
            print(f"{variant}: evaluated {n + 1}/{len(rows)}")
    if not results:
        raise ValueError("No held-out questions found.")
    length_words = [len(x["attempts"][0]["response"].split()) for x in results]
    summary = {
        "model": variant, "split": split, "n_questions": len(rows), "k": k,
        "pass_at_1": sum(row["pass1"] for row in results) / len(rows),
        "pass_at_k": sum(row["passk"] for row in results) / len(rows),
        "mean_reward": statistics.mean(row["attempts"][0]["reward"] for row in results),
        "mean_response_words": statistics.mean(length_words),
        "unique_recommendation_rate": len({r["selected_title"] for r in results if r["selected_title"]}) / max(1, sum(bool(r["selected_title"]) for r in results)),
        "format_rate": statistics.mean(row["attempts"][0]["format_ok"] for row in results),
    }
    destination = Path("outputs") / "evaluations" / f"{variant}_{split}{output_suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", destination)
    return summary
