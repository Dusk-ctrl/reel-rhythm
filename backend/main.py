"""Command-line entry point for the CSCI 5350 movie & music experiment.

Run from the backend/ folder, e.g.:
    python main.py --mode prepare
    python main.py --mode reward-test
    python main.py --mode train-sft
    python main.py --mode train-grpo
    python main.py --mode evaluate --model rl --split test
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=[
        "hardware", "prepare", "reward-test", "train-sft", "train-grpo",
        "evaluate", "checkpoints", "mistakes", "plot",
    ])
    p.add_argument("--movie-dir", default="data/imdb")
    p.add_argument("--music-dir", default="data/spotify")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--model", choices=["base", "sft", "rl"], default="base")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=40)
    a = p.parse_args()

    if a.mode == "hardware":
        from src.hardware import gpu_report
        print(json.dumps(gpu_report(), indent=2))
    elif a.mode == "prepare":
        from src.data_prep import prepare
        prepare(a.movie_dir, a.music_dir, a.output)
    elif a.mode == "reward-test":
        from src.reward import run_self_test
        run_self_test()
    elif a.mode == "train-sft":
        from src.train_sft import train_sft
        train_sft(max_steps=a.max_steps)
    elif a.mode == "train-grpo":
        from src.train_grpo import train_grpo
        train_grpo(max_steps=a.max_steps)
    elif a.mode == "evaluate":
        from src.evaluate import evaluate_model
        evaluate_model(a.model, k=a.k, split=a.split)
    elif a.mode == "checkpoints":
        import sys
        from src import evaluate_checkpoints
        model = a.model if a.model != "base" else "rl"
        sys.argv = ["evaluate_checkpoints", "--model", model, "--split", a.split, "--k", str(a.k)]
        evaluate_checkpoints.main()
    elif a.mode == "mistakes":
        from src.mistake_analysis import main as run
        run()
    elif a.mode == "plot":
        from src.plot_results import plot_all
        plot_all()


if __name__ == "__main__":
    main()
