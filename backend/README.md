# Python backend (CSCI 5350, Option A)

This folder holds the research code behind the movie & music recommender. It is
**separate from the Netlify site**: Netlify does not run any of this Python.

- `src/data_prep.py` turns real IMDb and Spotify Kaggle CSVs into multiple-choice questions whose answers are computed by plain Python, with disjoint train/validation/test/demo title splits.
- `src/reward.py` is the exact, verifiable reward (+1.0 correct, +0.2 strict format).
- `src/train_sft.py` and `src/train_grpo.py` fine-tune `google/gemma-4-E4B-it` with LoRA (SFT baseline vs. GRPO RL). They need a CUDA GPU with at least 48 GiB of memory.
- `src/evaluate.py`, `src/evaluate_checkpoints.py`, `src/mistake_analysis.py`, and `src/plot_results.py` score held-out splits (pass@1 / pass@k), find corrected mistakes, and plot real logs.
- `src/api.py` is a small FastAPI **rule-based** demo (not the trained model) that serves picks from the reserved demo split.

## Usage

Run everything from this `backend/` folder so `src.*` imports resolve:

```bash
pip install -r requirements.txt
# place CSVs in data/imdb/ and data/spotify/
python main.py --mode prepare
python main.py --mode reward-test
python main.py --mode train-sft        # GPU only
python main.py --mode train-grpo       # GPU only
python main.py --mode evaluate --model rl --split test
python main.py --mode plot
uvicorn src.api:app --port 8000        # demo API
```

Set `API_ACCESS_TOKEN` to require `Authorization: Bearer <token>` on `/api/recommend`.
