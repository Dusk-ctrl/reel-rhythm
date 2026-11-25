import pandas as pd
import json
import re
from collections import Counter


# -----------------------------
# Settings
# -----------------------------

MOVIES_FILE = "movies_clean.csv"

TRAIN_FILE = "movie_train.jsonl"
TEST_FILE = "movie_test.jsonl"

MIN_RATING = 7.0
MIN_VOTES = 1000
MAX_YEAR = 2025

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller"
]


# -----------------------------
# Load movie database
# -----------------------------

movies = pd.read_csv(MOVIES_FILE)

movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
movies["rating"] = pd.to_numeric(movies["rating"], errors="coerce")
movies["votes"] = pd.to_numeric(movies["votes"], errors="coerce")

movies = movies.dropna(
    subset=["title", "year", "rating", "votes"]
)

movies = movies[
    (movies["rating"] >= MIN_RATING) &
    (movies["votes"] >= MIN_VOTES) &
    (movies["year"] <= MAX_YEAR)
].copy()

movies["year"] = movies["year"].astype(int)

print("Reference movies loaded:", len(movies))


# -----------------------------
# Helper functions
# -----------------------------

def normalize(text):
    return str(text).strip().lower()


def parse_answer(answer):
    """
    Extract individual recommendations from the answer.
    """

    pattern = (
        r"^\s*\d+\.\s+(.+?)\s+\((\d{4})\)\s+-\s+"
        r"(.*?)\s+-\s+IMDb:\s*(\d+(?:\.\d+)?)\s*$"
    )

    results = []

    for line in answer.splitlines():

        match = re.match(pattern, line)

        if match is None:
            continue

        results.append({
            "title": match.group(1).strip(),
            "year": int(match.group(2)),
            "genres": [
                g.strip().lower()
                for g in match.group(3).split(",")
                if g.strip()
            ],
            "rating": float(match.group(4))
        })

    return results


def find_genre(question):
    question = question.lower()

    for genre in GENRES:
        if genre.lower() in question:
            return genre.lower()

    return None


def find_rating(question):
    """
    Find an explicitly stated minimum rating.

    Returns:
        (rating, operator)
    """

    patterns = [
        (r"at least\s+(\d+(?:\.\d+)?)", ">="),
        (r"or higher\s*(?:than)?\s*(\d+(?:\.\d+)?)", ">="),
        (r"rating of\s+(\d+(?:\.\d+)?)", ">="),
        (r"rated\s+(\d+(?:\.\d+)?)\s*\+", ">="),
        (r"above\s+(\d+(?:\.\d+)?)", ">")
    ]

    for pattern, operator in patterns:
        match = re.search(pattern, question.lower())

        if match:
            return float(match.group(1)), operator

    return None, None


def find_year_range(question):
    question = question.lower()

    # Example: from the 1990s
    match = re.search(r"\b((?:19|20)\d0)s\b", question)

    if match:
        start = int(match.group(1))
        return start, start + 9

    # Example: between 1980 and 1989
    match = re.search(
        r"(?:between|from)\s+(\d{4})\s+(?:and|to|-)\s+(\d{4})",
        question
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def find_expected_count(question):
    match = re.search(r"\b(3|5|10)\b", question)

    if match:
        return int(match.group(1))

    return None


# -----------------------------
# Validate one example
# -----------------------------

def validate_example(example):

    errors = []
    warnings = []

    messages = example.get("messages", [])

    if len(messages) < 2:
        return ["Invalid message structure"], warnings

    question = messages[0].get("content", "")
    answer = messages[1].get("content", "")

    recommendations = parse_answer(answer)

    if not recommendations:
        return ["Could not parse recommendations"], warnings

    # Check requested count

    expected_count = find_expected_count(question)

    if expected_count is not None:

        if len(recommendations) != expected_count:
            errors.append(
                f"Expected {expected_count} recommendations, "
                f"got {len(recommendations)}"
            )

    else:
        warnings.append("Question does not specify a recommendation count")

    # Check duplicates

    titles = [
        normalize(movie["title"])
        for movie in recommendations
    ]

    if len(titles) != len(set(titles)):
        errors.append("Duplicate movie titles")

    # Extract question constraints

    requested_genre = find_genre(question)

    requested_rating, rating_operator = find_rating(question)

    start_year, end_year = find_year_range(question)

    # Check every recommendation

    for movie in recommendations:

        title = normalize(movie["title"])

        year = movie["year"]

        rating = movie["rating"]

        # Find matching reference movie

        matching = movies[
            (movies["title"].map(normalize) == title) &
            (movies["year"] == year)
        ]

        if matching.empty:
            errors.append(
                f"Movie not found in database: {movie['title']} ({year})"
            )
            continue

        # Check whether the output metadata matches
        # at least one matching database entry.

        valid_record = False

        for _, reference in matching.iterrows():

            reference_genres = [
                g.strip().lower()
                for g in str(reference["genres"]).split(",")
                if g.strip()
            ]

            if (
                abs(float(reference["rating"]) - rating) < 0.11
                and set(reference_genres) == set(movie["genres"])
            ):
                valid_record = True
                break

        if not valid_record:
            errors.append(
                f"Incorrect metadata: {movie['title']}"
            )

        # Check requested genre

        if requested_genre is not None:

            if requested_genre not in movie["genres"]:
                errors.append(
                    f"Wrong genre: {movie['title']}"
                )

        # Check requested rating

        if requested_rating is not None:

            if rating_operator == ">=" and rating < requested_rating:
                errors.append(
                    f"Rating below requested minimum: {movie['title']}"
                )

            elif rating_operator == ">" and rating <= requested_rating:
                errors.append(
                    f"Rating not above requested minimum: {movie['title']}"
                )

        # Check requested year range

        if start_year is not None:

            if not (start_year <= year <= end_year):
                errors.append(
                    f"Year outside requested range: {movie['title']}"
                )

    return errors, warnings


# -----------------------------
# Validate train and test sets
# -----------------------------

def validate_file(filename):

    total = 0
    passed = 0
    failed = 0
    warning_count = 0

    all_errors = Counter()
    examples_with_errors = []

    with open(filename, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            total += 1

            try:
                example = json.loads(line)

            except json.JSONDecodeError:
                failed += 1
                all_errors["Invalid JSON"] += 1
                continue

            errors, warnings = validate_example(example)

            warning_count += len(warnings)

            if errors:

                failed += 1

                for error in errors:
                    all_errors[error] += 1

                if len(examples_with_errors) < 5:
                    examples_with_errors.append(
                        (line_number, errors)
                    )

            else:
                passed += 1

    print()
    print("=" * 45)
    print("FILE:", filename)
    print("=" * 45)

    print("Total examples:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Warnings:", warning_count)

    if all_errors:

        print("\nMost common errors:")

        for error, count in all_errors.most_common(10):
            print(f"{count}x - {error}")

    if examples_with_errors:

        print("\nExample failures:")

        for line_number, errors in examples_with_errors:

            print(f"\nLine {line_number}:")

            for error in errors:
                print(" -", error)

    return passed, failed


# -----------------------------
# Run validation
# -----------------------------

train_passed, train_failed = validate_file(TRAIN_FILE)

test_passed, test_failed = validate_file(TEST_FILE)

print()
print("=" * 45)
print("FINAL RESULTS")
print("=" * 45)

print("Training examples passed:", train_passed)
print("Training examples failed:", train_failed)

print("Testing examples passed:", test_passed)
print("Testing examples failed:", test_failed)
