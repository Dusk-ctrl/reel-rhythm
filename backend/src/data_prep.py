"""Turn the real IMDb and Spotify CSV files into checkable questions.

I deliberately use the columns already present in each dataset rather than
asking an AI model to invent titles or ratings. -- Sandipta, CSCI 5350
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# The same project can use different Kaggle exports without renaming every CSV.
ALIASES = {
    "movie": {
        "title": ["Series_Title", "movie_title", "Movie_Title", "primaryTitle", "title", "name"],
        "score": ["IMDB_Rating", "imdb_score", "averageRating", "avg_vote", "rating", "vote_average"],
        "genre": ["genre", "genres", "main_genre", "listed_in"],
    },
    "music": {
        "title": ["track_name", "song_name", "track", "song", "title", "name"],
        "score": ["track_popularity", "spotify_popularity", "popularity", "popularity_score"],
        "genre": ["artist_genres", "track_genre", "genre", "genres"],
        "artist": ["artist_name", "artist", "artists", "track_artist"],
    },
}


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def choose_column(columns, aliases):
    lookup = {normalized(col): col for col in columns}
    return next((lookup[normalized(alias)] for alias in aliases if normalized(alias) in lookup), None)


def parse_genres(value):
    """Convert values like 'Action, Drama' or "['pop', 'rock']" into words."""
    if value is None:
        return []
    return sorted(set(
        part.strip().strip("[]'\" ").lower()
        for part in re.split(r"[,|;/]", str(value))
        if part.strip().strip("[]'\" ").lower() not in ("", "nan", "none", "unknown")
    ))


def load_catalog(directory, kind):
    """Read each usable CSV and explain when a file cannot be used."""
    import pandas as pd  # Only needed during data preparation, not reward tests.

    folder = Path(directory)
    paths = sorted(folder.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files were found inside {folder}. Download and extract the {kind} dataset first."
        )

    cleaned = []
    problems = []
    for path in paths:
        try:
            frame = pd.read_csv(path, nrows=200_000, encoding="utf-8-sig", low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            problems.append(f"{path.name}: cannot read CSV ({error})")
            continue
        columns = {key: choose_column(frame.columns, choices) for key, choices in ALIASES[kind].items()}
        if not columns["title"] or not columns["score"]:
            problems.append(f"{path.name}: columns {list(frame.columns)} lack a title and numeric score")
            continue
        print(f"Reading {kind}: {path.name} (title={columns['title']}, score={columns['score']})")
        for _, row in frame.iterrows():
            title = str(row[columns["title"]]).strip()
            raw_score = str(row[columns["score"]]).replace(",", "").strip()
            try:
                score = float(raw_score)
            except ValueError:
                continue
            if not (2 <= len(title) <= 120) or title.lower() in ("nan", "none", "unknown"):
                continue
            if kind == "movie" and not 0 <= score <= 10:
                continue
            if kind == "music" and not 0 <= score <= 100:
                continue
            raw_genre = row[columns["genre"]] if columns.get("genre") else None
            artist = str(row[columns["artist"]]).strip() if columns.get("artist") else ""
            if artist.lower() == "nan":
                artist = ""
            # Movies need a genre, because the professor specifically wants genre rules.
            genres = parse_genres(raw_genre)
            if kind == "movie" and not genres:
                continue
            cleaned.append({
                "title": title,
                "score": score,
                "genres": genres,
                "artist": artist,
                "kind": kind,
            })

    # Multiple Kaggle CSVs sometimes repeat the same title or track.
    distinct = {}
    for item in cleaned:
        key = (normalized(item["title"]), normalized(item["artist"]))
        if key not in distinct or item["score"] > distinct[key]["score"]:
            distinct[key] = item
    records = sorted(distinct.values(), key=lambda x: (x["title"].lower(), x["artist"].lower()))
    if len(records) < 80:
        raise ValueError(
            f"Only {len(records)} usable {kind} records were found; at least 80 are needed for the reserved demo split.\n"
            + "\n".join(problems)
            + "\nMovies need title, numeric IMDb rating, and genre. Music needs title and track popularity."
        )
    print(f"Kept {len(records)} unique {kind} records.\n")
    return records


def split_records(records, seed):
    """Split titles BEFORE creating questions so train and test titles never overlap."""
    copied = list(records)
    random.Random(seed).shuffle(copied)
    n = len(copied)
    a, b, c = int(n * .70), int(n * .85), int(n * .95)
    return {"train": copied[:a], "validation": copied[a:b], "test": copied[b:c], "demo": copied[c:]}


def question_for_candidates(candidates, kind, requirement, question_number):
    """Calculate the answer with ordinary Python, never with Gemma."""
    if kind == "movie":
        genre = requirement
        eligible = [item for item in candidates if genre in item["genres"] and item["score"] >= 7.0]
        instruction = f"Choose the highest IMDb-rated {genre} movie rated at least 7.0/10."
        score_name = "IMDb rating (0-10)"
    else:
        genre = requirement
        eligible = [item for item in candidates if item["score"] >= 60 and
                    (genre is None or genre in item["genres"])]
        extra = f" with genre {genre}" if genre else ""
        instruction = f"Choose the highest-popularity Spotify track{extra} with popularity at least 60/100."
        score_name = "Spotify popularity (0-100)"

    if not eligible:
        return None
    best_score = max(item["score"] for item in eligible)
    winners = [item for item in eligible if item["score"] == best_score]
    if len(winners) != 1:  # Skip tied questions rather than inventing a correct answer.
        return None
    winning_item = winners[0]
    winning_index = next(i for i, item in enumerate(candidates) if item is winning_item)
    right_letter = "ABCD"[winning_index]
    display = []
    for letter, item in zip("ABCD", candidates):
        fields = [item["title"]]
        if item["artist"]:
            fields.append(f"artist: {item['artist']}")
        if item["genres"]:
            fields.append(f"genres: {', '.join(item['genres'])}")
        fields.append(f"{score_name}: {item['score']:g}")
        display.append(f"{letter}. " + " | ".join(fields))
    prompt = (
        instruction + " Only choose one of the four dataset-listed options.\n"
        + "\n".join(display)
        + "\nBriefly explain the deciding condition inside <think>...</think>, "
        + "and give ONE letter inside <answer>...</answer>."
    )
    explanation = (
        f"I checked the genre and minimum score, then compared the eligible options. "
        f"Choice {right_letter} has the highest eligible score, {best_score:g}."
    )
    reference = f"<think>{explanation}</think><answer>{right_letter}</answer>"
    return {
        "id": f"{kind}-{question_number:05d}", "category": kind,
        "prompt": prompt, "ground_truth": right_letter,
        "reference": reference, "expected_title": winning_item["title"],
        "candidate_titles": [item["title"] for item in candidates],
        "correctness_rule": instruction,
    }


def build_questions(records, kind, maximum, seed):
    """Sample a range of qualifying titles instead of always taking global top 10."""
    rng = random.Random(seed)
    high = [item for item in records if item["score"] >= (7.0 if kind == "movie" else 60)]
    if not high:
        raise ValueError(f"No {kind} entries meet the minimum score (7.0 or 60).")
    rng.shuffle(high)
    questions = []
    attempted = 0
    for target in high:
        if len(questions) >= maximum:
            break
        for _ in range(12):
            distractors = rng.sample(records, 3)
            if target in distractors:
                continue
            choices = [target] + distractors
            if kind == "movie":
                genre = rng.choice(target["genres"])
            else:
                genre = rng.choice(target["genres"]) if target["genres"] and rng.random() < .5 else None
            eligible = [item for item in choices if item["score"] >= (7 if kind == "movie" else 60) and
                        (genre is None or genre in item["genres"])]
            if not eligible or len([x for x in eligible if x["score"] == max(v["score"] for v in eligible)]) != 1:
                continue
            # Rotate the correct letter to avoid an A/B/C/D shortcut.
            winner = max(eligible, key=lambda item: item["score"])
            others = [item for item in choices if item is not winner]
            rng.shuffle(others)
            position = len(questions) % 4
            others.insert(position, winner)
            row = question_for_candidates(others, kind, genre, attempted)
            attempted += 1
            if row:
                questions.append(row)
                break
    if not questions:
        raise ValueError(f"Could not construct any unambiguous {kind} questions.")
    return questions


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prepare(movie_dir="data/imdb", music_dir="data/spotify", output="data/processed"):
    """Prepare disjoint splits; validation/test are NEVER input to training."""
    sources = {
        "movie": split_records(load_catalog(movie_dir, "movie"), 42),
        "music": split_records(load_catalog(music_dir, "music"), 83),
    }
    limits = {"train": 600, "validation": 120, "test": 120, "demo": 50}
    split_rows = {}
    for split in ["train", "validation", "test", "demo"]:
        rows = []
        for kind in ("movie", "music"):
            records = sources[kind][split]
            if len(records) < 4:
                raise ValueError(f"Not enough unique {kind} records in the {split} partition.")
            rows += build_questions(records, kind, limits[split], seed={"train": 11, "validation": 22, "test": 33, "demo": 44}[split] + (1 if kind == "movie" else 2))
        random.Random(200).shuffle(rows)
        split_rows[split] = rows
        write_jsonl(Path(output) / f"{split}.jsonl", rows)
        print(f"{split:10} -> {len(rows)} questions | correct labels: {dict(Counter(row['ground_truth'] for row in rows))}")

    # Verify that an individual title cannot appear in both train and test prompts.
    title_sets = {name: set((row["category"], t.casefold()) for row in rows for t in row["candidate_titles"])
                  for name, rows in split_rows.items()}
    for a in title_sets:
        for b in title_sets:
            if a < b and title_sets[a].intersection(title_sets[b]):
                raise AssertionError(f"Entity leakage between {a} and {b}")
    summary = {
        "seed": 42,
        "split_method": "disjoint movie/track titles before generating questions",
        "counts": {key: len(value) for key, value in split_rows.items()},
        "evaluation_warning": "TEST is for final scoring; do not train or tune on it",
    }
    output_path = Path(output)
    (output_path / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return split_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movie-dir", default="data/imdb")
    p.add_argument("--music-dir", default="data/spotify")
    p.add_argument("--output", default="data/processed")
    a = p.parse_args()
    prepare(a.movie_dir, a.music_dir, a.output)


if __name__ == "__main__":
    main()
