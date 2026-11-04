"""Create graphs only from real saved training logs and evaluation runs."""
import json
from pathlib import Path


def plot_all():
    import matplotlib.pyplot as plt
    root = Path("outputs")
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for variant, label in [("sft", "SFT"), ("rl", "GRPO")]:
        filename = root / variant / "log_history.json"
        if not filename.exists():
            print(f"Skipped {label}: training has not run yet ({filename} missing)")
            continue
        data = json.loads(filename.read_text(encoding="utf-8"))
        values = [(item["step"], item["reward"]) for item in data if "reward" in item and "step" in item]
        if not values:
            print(f"Skipped {label} reward curve: no reward points in trainer logs")
            continue
        plt.figure()
        plt.plot([x for x, _ in values], [y for _, y in values], marker="o")
        plt.xlabel("Training step")
        plt.ylabel("Average sampled reward")
        plt.title(f"{label} training reward (actual logs)")
        plt.tight_layout()
        plt.savefig(plots / f"{variant}_reward.png", dpi=150)
        plt.close()

    # Plot actual checkpoint evaluation (not placeholder percentages).
    for variant in ("sft", "rl"):
        for split in ("validation", "test"):
            path = root / "evaluations" / f"{variant}_{split}_checkpoints.json"
            if not path.exists():
                continue
            checkpoints = json.loads(path.read_text(encoding="utf-8"))
            if not checkpoints:
                continue
            for key, ylabel, ending in (("pass_at_1", "pass@1 accuracy", "accuracy"),
                                         ("mean_response_words", "Average response length (words)", "length")):
                plt.figure()
                plt.plot([r["step"] for r in checkpoints], [r[key] for r in checkpoints], marker="o")
                plt.xlabel("Training step")
                plt.ylabel(ylabel)
                plt.title(f"{variant.upper()} {split}: {ylabel} over training")
                plt.tight_layout()
                plt.savefig(plots / f"{variant}_{split}_{ending}.png", dpi=150)
                plt.close()

    # Held-out accuracy across checkpoints requires saved intermediate models.
    # A single final checkpoint is NOT a test-accuracy learning curve.
    scores = []
    for variant in ("base", "sft", "rl"):
        file = root / "evaluations" / f"{variant}_test.json"
        if file.exists():
            scores.append(json.loads(file.read_text(encoding="utf-8"))["summary"])
    if len(scores) == 3:
        plt.figure()
        plt.bar([r["model"] for r in scores], [r["pass_at_1"] for r in scores])
        plt.ylabel("Held-out pass@1")
        plt.ylim(0, 1)
        plt.title("Same unseen test set")
        plt.tight_layout()
        plt.savefig(plots / "model_comparison.png", dpi=150)
        plt.close()
    else:
        print("Model comparison plot requires real base, SFT, and RL evaluation files.")
    print("Plots written to", plots)
