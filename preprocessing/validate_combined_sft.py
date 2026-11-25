import pandas as pd
import json
import re
import ast


# -----------------------------
# SETTINGS
# -----------------------------

MOVIE_FILE = "movies_clean.csv"
MUSIC_FILE = "spotify_clean.csv"

TRAIN_FILE = "combined_train.jsonl"
TEST_FILE = "combined_test.jsonl"


# -----------------------------
# LOAD MOVIES
# -----------------------------

movies = pd.read_csv(MOVIE_FILE)

movies["year"] = pd.to_numeric(
    movies["year"], errors="coerce"
)

movies["rating"] = pd.to_numeric(
    movies["rating"], errors="coerce"
)

movies["votes"] = pd.to_numeric(
    movies["votes"], errors="coerce"
)

movies = movies.dropna(
    subset=["id", "title", "year", "genres", "rating", "votes"]
)

movies["id"] = movies["id"].astype(str)

movie_lookup = {}

for _, movie in movies.iterrows():
    movie_lookup[movie["id"]] = movie


# -----------------------------
# LOAD MUSIC
# -----------------------------

music = pd.read_csv(MUSIC_FILE)

music["popularity"] = pd.to_numeric(
    music["popularity"], errors="coerce"
)

music["duration_min"] = pd.to_numeric(
    music["duration_min"], errors="coerce"
)

music["release_date"] = pd.to_datetime(
    music["release_date"], errors="coerce"
)

music["year"] = music["release_date"].dt.year

music = music.dropna(
    subset=[
        "id", "title", "artist", "popularity",
        "year", "duration_min", "explicit", "genres"
    ]
).copy()

music["id"] = music["id"].astype(str)

music["year"] = music["year"].astype(int)

track_lookup = {}

for _, track in music.iterrows():
    track_lookup[track["id"]] = track


# -----------------------------
# GENRE FUNCTIONS
# -----------------------------

def get_genres(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return [
                str(genre).strip().lower()
                for genre in parsed
            ]

    except (ValueError, SyntaxError):
        pass

    return [
        genre.strip().lower()
        for genre in str(value).split(",")
        if genre.strip()
    ]


# -----------------------------
# PARSE MOVIE RECOMMENDATIONS
# -----------------------------

def parse_movies(response):
    recommendations = []

    pattern = (
        r"^\s*(\d+)\.\s+(.+?)\s+\((\d{4})\)"
        r"\s+\|\s+Genres:\s*(.*?)"
        r"\s+\|\s+Rating:\s*([\d.]+)"
        r"\s+\|\s+Votes:\s*(\d+)"
        r"\s+\|\s+IMDb ID:\s*(tt\d+)\s*$"
    )

    for line in response.splitlines():

        match = re.search(pattern, line)

        if match:
            recommendations.append({
                "number": int(match.group(1)),
                "title": match.group(2).strip(),
                "year": int(match.group(3)),
                "genres": match.group(4).strip(),
                "rating": float(match.group(5)),
                "votes": int(match.group(6)),
                "id": match.group(7)
            })

    return recommendations


# -----------------------------
# PARSE MUSIC RECOMMENDATIONS
# -----------------------------

def parse_music(response):
    recommendations = []

    pattern = (
        r"^\s*(\d+)\.\s+(.+?)\s+\|\s+Genres:\s*(.*?)"
        r"\s+\|\s+Popularity:\s*(\d+)"
        r"\s+\|\s+Year:\s*(\d+)"
        r"\s+\|\s+Duration:\s*([\d.]+)\s*min"
        r"\s+\|\s+Explicit:\s*(Yes|No)"
        r"\s+\|\s+Spotify ID:\s*([A-Za-z0-9]+)\s*$"
    )

    for line in response.splitlines():

        match = re.search(pattern, line)

        if match:
            display_name = match.group(2).strip()

            recommendations.append({
                "number": int(match.group(1)),
                "display_name": display_name,
                "genres": match.group(3).strip(),
                "popularity": int(match.group(4)),
                "year": int(match.group(5)),
                "duration": float(match.group(6)),
                "explicit": match.group(7),
                "id": match.group(8)
            })

    return recommendations


# -----------------------------
# VALIDATE MOVIES
# -----------------------------

def validate_movies(recommendations, errors):

    seen_ids = set()

    for movie in recommendations:

        movie_id = movie["id"]

        if movie_id in seen_ids:
            errors.append(
                f"Duplicate movie ID: {movie_id}"
            )

        seen_ids.add(movie_id)

        if movie_id not in movie_lookup:
            errors.append(
                f"Movie ID does not exist: {movie_id}"
            )
            continue

        source = movie_lookup[movie_id]

        if movie["title"].lower() != str(source["title"]).strip().lower():
            errors.append(
                f"Movie title mismatch: {movie_id}"
            )

        if movie["year"] != int(source["year"]):
            errors.append(
                f"Movie year mismatch: {movie_id}"
            )

        if movie["rating"] != round(float(source["rating"]), 1):
            errors.append(
                f"Movie rating mismatch: {movie_id}"
            )

        if movie["votes"] != int(source["votes"]):
            errors.append(
                f"Movie votes mismatch: {movie_id}"
            )

        expected_genres = get_genres(source["genres"])
        actual_genres = get_genres(movie["genres"])

        if set(expected_genres) != set(actual_genres):
            errors.append(
                f"Movie genres mismatch: {movie_id}"
            )


# -----------------------------
# VALIDATE MUSIC
# -----------------------------

def validate_music(recommendations, errors):

    seen_ids = set()

    for track in recommendations:

        track_id = track["id"]

        if track_id in seen_ids:
            errors.append(
                f"Duplicate Spotify ID: {track_id}"
            )

        seen_ids.add(track_id)

        if track_id not in track_lookup:
            errors.append(
                f"Spotify ID does not exist: {track_id}"
            )
            continue

        source = track_lookup[track_id]

        expected_name = (
            f"{str(source['title']).strip()} - "
            f"{str(source['artist']).strip()}"
        )

        if track["display_name"].lower() != expected_name.lower():
            errors.append(
                f"Track title/artist mismatch: {track_id}"
            )

        if track["popularity"] != int(source["popularity"]):
            errors.append(
                f"Track popularity mismatch: {track_id}"
            )

        if track["year"] != int(source["year"]):
            errors.append(
                f"Track year mismatch: {track_id}"
            )

        if abs(track["duration"] - float(source["duration_min"])) > 0.011:
            errors.append(
                f"Track duration mismatch: {track_id}"
            )

        source_explicit = str(source["explicit"]).strip().lower()

        expected_explicit = (
            "Yes"
            if source_explicit in ["true", "1", "yes"]
            else "No"
        )

        if track["explicit"] != expected_explicit:
            errors.append(
                f"Track explicit mismatch: {track_id}"
            )

        expected_genres = get_genres(source["genres"])
        actual_genres = get_genres(track["genres"])

        if set(expected_genres) != set(actual_genres):
            errors.append(
                f"Track genres mismatch: {track_id}"
            )


# -----------------------------
# VALIDATE REQUESTED COUNTS
# -----------------------------



def validate_counts(question, movie_recommendations,
                    music_recommendations, errors):

    # Find the movie count before "movies"
    movie_match = re.search(
        r"\b(\d+)\s+[\w-]+\s+movies?\b",
        question,
        re.IGNORECASE
    )

    # Find the music count after "and" and before "tracks"
    music_match = re.search(
        r"\band\s+(\d+)\b",
        question,
        re.IGNORECASE
    )

    if movie_match:
        expected_movies = int(movie_match.group(1))

        if len(movie_recommendations) != expected_movies:
            errors.append(
                f"Expected {expected_movies} movies, "
                f"got {len(movie_recommendations)}"
            )
    else:
        errors.append(
            "Could not identify requested movie count."
        )

    if music_match:
        expected_tracks = int(music_match.group(1))

        if len(music_recommendations) != expected_tracks:
            errors.append(
                f"Expected {expected_tracks} tracks, "
                f"got {len(music_recommendations)}"
            )
    else:
        errors.append(
            "Could not identify requested music count."
        )




# -----------------------------
# VALIDATE ONE EXAMPLE
# -----------------------------

def validate_example(example):

    errors = []

    question = example["messages"][0]["content"]
    response = example["messages"][1]["content"]

    movie_recommendations = parse_movies(response)
    music_recommendations = parse_music(response)

    if not movie_recommendations:
        errors.append("No movie recommendations could be parsed.")

    if not music_recommendations:
        errors.append("No music recommendations could be parsed.")

    # Check recommendation numbering
    movie_numbers = [
        movie["number"] for movie in movie_recommendations
    ]

    music_numbers = [
        track["number"] for track in music_recommendations
    ]

    if movie_numbers != list(range(1, len(movie_numbers) + 1)):
        errors.append("Movie numbering is incorrect.")

    if music_numbers != list(range(1, len(music_numbers) + 1)):
        errors.append("Music numbering is incorrect.")

    validate_movies(movie_recommendations, errors)
    validate_music(music_recommendations, errors)

    validate_counts(
        question,
        movie_recommendations,
        music_recommendations,
        errors
    )

    return errors


# -----------------------------
# VALIDATE JSONL FILE
# -----------------------------

def validate_file(filename):

    total = 0
    passed = 0
    failed = 0

    print("\nValidating:", filename)

    with open(filename, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            total += 1

            try:
                example = json.loads(line)

                errors = validate_example(example)

            except Exception as error:
                errors = [f"Could not read example: {error}"]

            if errors:
                failed += 1

                print(f"\nExample {line_number} FAILED:")

                for error in errors:
                    print("  -", error)

            else:
                passed += 1

    print("\nRESULTS")
    print("Total:", total)
    print("Passed:", passed)
    print("Failed:", failed)

    return total, passed, failed


# -----------------------------
# RUN VALIDATION
# -----------------------------

print("Reference movies loaded:", len(movie_lookup))
print("Reference tracks loaded:", len(track_lookup))

train_results = validate_file(TRAIN_FILE)
test_results = validate_file(TEST_FILE)

total = train_results[0] + test_results[0]
passed = train_results[1] + test_results[1]
failed = train_results[2] + test_results[2]

print("\n==============================")
print("FINAL VALIDATION")
print("==============================")

print("Total:", total)
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\nVALIDATION PASSED!")
else:
    print("\nVALIDATION FAILED: Check the errors above.")
