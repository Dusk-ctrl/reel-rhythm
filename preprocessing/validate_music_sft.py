import pandas as pd
import json
import re
import ast


# =========================
# SETTINGS
# =========================

INPUT_FILE = "spotify_clean.csv"
TRAIN_FILE = "music_train.jsonl"
TEST_FILE = "music_test.jsonl"

# =========================
# LOAD REFERENCE DATA
# =========================

df = pd.read_csv(INPUT_FILE)

df["id"] = df["id"].astype(str)
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
df["artist_popularity"] = pd.to_numeric(
    df["artist_popularity"], errors="coerce"
)
df["duration_min"] = pd.to_numeric(
    df["duration_min"], errors="coerce"
)

df["release_date"] = pd.to_datetime(
    df["release_date"], errors="coerce"
)
df["year"] = df["release_date"].dt.year

# Convert explicit values into True/False
def convert_explicit(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return True

    if value in ["false", "0", "no"]:
        return False

    return None


df["explicit"] = df["explicit"].apply(convert_explicit)


# Create a lookup using Spotify IDs
track_lookup = {}

for _, track in df.iterrows():
    track_lookup[track["id"]] = track


# =========================
# GENRE HANDLING
# =========================

def parse_genres(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Handle Python-list-like strings
    if value.startswith("["):
        try:
            result = ast.literal_eval(value)

            if isinstance(result, list):
                return [
                    str(g).strip().lower()
                    for g in result
                    if str(g).strip()
                ]
        except (ValueError, SyntaxError):
            pass

    # Handle comma-separated genres
    return [
        g.strip().lower()
        for g in value.split(",")
        if g.strip()
    ]


# Find genres that appear in the question
available_genres = set()

for value in df["genres"].dropna():
    for genre in parse_genres(value):
        available_genres.add(genre)


def find_question_genre(question):
    question = question.lower()

    # Check longer genres first to avoid matching
    # a short genre inside a longer genre.
    sorted_genres = sorted(
        available_genres,
        key=len,
        reverse=True
    )

    for genre in sorted_genres:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(genre)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, question):
            return genre

    return None


# =========================
# PARSE RECOMMENDATIONS
# =========================

def parse_recommendations(response):
    recommendations = []

    pattern = re.compile(
        r"^\s*(\d+)\.\s+(.+)\s+-\s+(.+?)"
        r"\s+\|\s+Genres:\s*(.*?)"
        r"\s+\|\s+Popularity:\s*(\d+)"
        r"\s+\|\s+Year:\s*(\d+)"
        r"\s+\|\s+Duration:\s*([\d.]+)\s*min"
        r"\s+\|\s+Explicit:\s*(Yes|No)"
        r"\s+\|\s+Spotify ID:\s*([A-Za-z0-9]+)\s*$",
        re.IGNORECASE
    )

    for line in response.splitlines():
        line = line.strip()

        if not line:
            continue

        match = pattern.match(line)

        if not match:
            recommendations.append({
                "line": line,
                "valid_format": False
            })
            continue

        recommendations.append({
            "number": int(match.group(1)),
            "title": match.group(2).strip(),
            "artist": match.group(3).strip(),
            "genres": [
                g.strip().lower()
                for g in match.group(4).split(",")
                if g.strip()
            ],
            "popularity": int(match.group(5)),
            "year": int(match.group(6)),
            "duration": float(match.group(7)),
            "explicit": match.group(8).lower() == "yes",
            "id": match.group(9),
            "valid_format": True
        })

    return recommendations


# =========================
# VALIDATE ONE EXAMPLE
# =========================

def validate_example(example, example_number):
    errors = []
    warnings = []

    messages = example.get("messages", [])

    if len(messages) < 2:
        return [
            f"Example {example_number}: Missing user or assistant message."
        ], warnings

    question = messages[0].get("content", "")
    response = messages[1].get("content", "")

    question_lower = question.lower()

    recommendations = parse_recommendations(response)

    if not recommendations:
        errors.append(
            f"Example {example_number}: No recommendations found."
        )
        return errors, warnings

    # Check requested recommendation count
    count_match = re.search(
        r"\b(\d+)\s+(?:tracks?|songs?|recommendations?)\b",
        question_lower
    )

    if count_match:
        expected_count = int(count_match.group(1))

        if len(recommendations) != expected_count:
            errors.append(
                f"Example {example_number}: "
                f"Expected {expected_count} tracks, "
                f"found {len(recommendations)}."
            )
    else:
        warnings.append(
            f"Example {example_number}: Could not identify requested count."
        )

    seen_ids = set()
    valid_tracks = []

    # Check every recommendation
    for recommendation in recommendations:

        if not recommendation["valid_format"]:
            errors.append(
                f"Example {example_number}: Invalid response format: "
                f"{recommendation['line']}"
            )
            continue

        track_id = recommendation["id"]

        if track_id in seen_ids:
            errors.append(
                f"Example {example_number}: Duplicate track ID {track_id}."
            )

        seen_ids.add(track_id)

        if track_id not in track_lookup:
            errors.append(
                f"Example {example_number}: "
                f"Spotify ID {track_id} does not exist."
            )
            continue

        source = track_lookup[track_id]
        valid_tracks.append((recommendation, source))

        # Check title
        if recommendation["title"].strip().lower() != str(
            source["title"]
        ).strip().lower():
            errors.append(
                f"Example {example_number}: Title mismatch for {track_id}."
            )

        # Check artist
        if recommendation["artist"].strip().lower() != str(
            source["artist"]
        ).strip().lower():
            errors.append(
                f"Example {example_number}: Artist mismatch for {track_id}."
            )

        # Check genres
        source_genres = parse_genres(source["genres"])

        if set(recommendation["genres"]) != set(source_genres):
            errors.append(
                f"Example {example_number}: Genre mismatch for {track_id}."
            )

        # Check popularity
        if pd.notna(source["popularity"]):
            if recommendation["popularity"] != round(
                source["popularity"]
            ):
                errors.append(
                    f"Example {example_number}: "
                    f"Popularity mismatch for {track_id}."
                )

        # Check year
        if pd.notna(source["year"]):
            if recommendation["year"] != int(source["year"]):
                errors.append(
                    f"Example {example_number}: Year mismatch for {track_id}."
                )

        # Check duration (allow rounding to 2 decimal places)
        if pd.notna(source["duration_min"]):
            if abs(
                recommendation["duration"]
                - source["duration_min"]
            ) > 0.011:
                errors.append(
                    f"Example {example_number}: "
                    f"Duration mismatch for {track_id}."
                )

        # Check explicit status
        if source["explicit"] is not None:
            if recommendation["explicit"] != source["explicit"]:
                errors.append(
                    f"Example {example_number}: "
                    f"Explicit status mismatch for {track_id}."
                )

    # =========================
    # CHECK QUESTION CONDITIONS
    # =========================

    # Genre condition
    question_genre = find_question_genre(question)

    if question_genre:
        for recommendation, source in valid_tracks:
            source_genres = parse_genres(source["genres"])

            if question_genre not in source_genres:
                errors.append(
                    f"Example {example_number}: "
                    f"{recommendation['title']} does not match "
                    f"genre '{question_genre}'."
                )

    # Duration condition: under X minutes
    duration_match = re.search(
        r"under\s+([\d.]+)\s+minutes?",
        question_lower
    )

    if duration_match:
        max_duration = float(duration_match.group(1))

        for recommendation, source in valid_tracks:
            if source["duration_min"] >= max_duration:
                errors.append(
                    f"Example {example_number}: "
                    f"{recommendation['title']} is not under "
                    f"{max_duration} minutes."
                )

    # Artist popularity condition
    artist_popularity_match = re.search(
        r"artist.{0,40}?popularity\s+of\s+at\s+least\s+(\d+)",
        question_lower
    )

    if artist_popularity_match:
        minimum = int(artist_popularity_match.group(1))

        for recommendation, source in valid_tracks:
            if pd.isna(source["artist_popularity"]):
                errors.append(
                    f"Example {example_number}: "
                    f"Missing artist popularity for {recommendation['title']}."
                )
            elif source["artist_popularity"] < minimum:
                errors.append(
                    f"Example {example_number}: "
                    f"{recommendation['title']} has artist popularity "
                    f"below {minimum}."
                )

    # Track popularity condition
    elif "popularity" in question_lower:
        popularity_match = re.search(
            r"popularity\s+of\s+at\s+least\s+(\d+)",
            question_lower
        )

        if popularity_match:
            minimum = int(popularity_match.group(1))

            for recommendation, source in valid_tracks:
                if source["popularity"] < minimum:
                    errors.append(
                        f"Example {example_number}: "
                        f"{recommendation['title']} has track popularity "
                        f"below {minimum}."
                    )

    # Decade condition
    decade_match = re.search(
        r"\b((?:19|20)\d)0s\b",
        question_lower
    )

    if decade_match:
        start_year = int(decade_match.group(1) + "0")
        end_year = start_year + 9

        for recommendation, source in valid_tracks:
            if pd.notna(source["year"]):
                if not start_year <= source["year"] <= end_year:
                    errors.append(
                        f"Example {example_number}: "
                        f"{recommendation['title']} is outside "
                        f"the {start_year}s."
                    )

    # Non-explicit condition
    if (
        "non-explicit" in question_lower
        or "non explicit" in question_lower
        or "without explicit" in question_lower
    ):
        for recommendation, source in valid_tracks:
            if source["explicit"] is True:
                errors.append(
                    f"Example {example_number}: "
                    f"{recommendation['title']} is explicit."
                )

    return errors, warnings


# =========================
# VALIDATE A JSONL FILE
# =========================

def validate_file(filename):
    print("\n" + "=" * 55)
    print("Validating:", filename)
    print("=" * 55)

    total = 0
    passed = 0
    failed = 0
    warning_count = 0

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            total += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                print(f"Example {total}: Invalid JSON.")
                continue

            errors, warnings = validate_example(example, total)

            if errors:
                failed += 1

                print(f"\nFAILED Example {total}:")

                for error in errors:
                    print("  ERROR:", error)

            else:
                passed += 1

            warning_count += len(warnings)

            for warning in warnings:
                print("  WARNING:", warning)

    print("\nRESULTS")
    print("-" * 40)
    print("Total:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Warnings:", warning_count)

    return failed


# =========================
# RUN VALIDATION
# =========================

print("Reference tracks loaded:", len(df))
print("Unique Spotify IDs:", df["id"].nunique())

train_failed = validate_file(TRAIN_FILE)
test_failed = validate_file(TEST_FILE)

print("\n" + "=" * 55)
print("FINAL VALIDATION")
print("=" * 55)

if train_failed == 0 and test_failed == 0:
    print("SUCCESS: Both music datasets passed validation.")
else:
    print("VALIDATION FAILED: Check the errors above.")
