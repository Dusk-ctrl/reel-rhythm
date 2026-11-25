import pandas as pd
import json
import random
import ast
from collections import Counter


# -----------------------------
# Settings
# -----------------------------

INPUT_FILE = "spotify_clean.csv"

TRAIN_FILE = "music_train.jsonl"
TEST_FILE = "music_test.jsonl"

RANDOM_SEED = 42
TEST_SIZE = 0.20

TARGET_EXAMPLES = 1000

MAX_YEAR = 2025
MIN_GENRE_TRACKS = 10

random.seed(RANDOM_SEED)


# -----------------------------
# Load dataset
# -----------------------------

spotify = pd.read_csv(INPUT_FILE)

print("Tracks loaded:", len(spotify))


# -----------------------------
# Basic cleanup
# -----------------------------

spotify["popularity"] = pd.to_numeric(
    spotify["popularity"],
    errors="coerce"
)

spotify["artist_popularity"] = pd.to_numeric(
    spotify["artist_popularity"],
    errors="coerce"
)

spotify["duration_min"] = pd.to_numeric(
    spotify["duration_min"],
    errors="coerce"
)

spotify["release_date"] = pd.to_datetime(
    spotify["release_date"],
    errors="coerce"
)

spotify["year"] = spotify["release_date"].dt.year


# Convert explicit values into True / False

def convert_explicit(value):

    value = str(value).strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    return None


spotify["explicit"] = spotify["explicit"].apply(
    convert_explicit
)

spotify["genres"] = spotify["genres"].fillna("")


# -----------------------------
# Filter tracks
# -----------------------------

spotify = spotify.dropna(
    subset=[
        "id",
        "title",
        "artist",
        "popularity",
        "year",
        "duration_min",
        "explicit"
    ]
)

spotify = spotify[
    (spotify["popularity"] >= 0) &
    (spotify["popularity"] <= 100) &
    (spotify["year"] <= MAX_YEAR) &
    (spotify["duration_min"] > 0) &
    (spotify["duration_min"] < 30)
].copy()

spotify["year"] = spotify["year"].astype(int)

print("Tracks after filtering:", len(spotify))


# -----------------------------
# Helper functions
# -----------------------------

def get_genres(track):
    """
    Convert the genre field into a list.

    Handles both:
    "pop, dance pop"

    and:
    "['pop', 'dance pop']"
    """

    value = str(track["genres"]).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(genre).strip().lower()
                for genre in parsed
                if str(genre).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [
        genre.strip().lower()
        for genre in value.split(",")
        if genre.strip()
    ]


def has_genre(track, genre):
    """
    Check whether a track belongs to a genre.
    """

    return genre.lower() in get_genres(track)


def format_track(track):
    """
    Format one Spotify track for the assistant's response.

    Spotify ID is included to make recommendations
    unambiguous and easier to verify later.
    """

    genres = ", ".join(get_genres(track))

    explicit = "Yes" if track["explicit"] else "No"

    return (
        f"{track['title']} - "
        f"{track['artist']} | "
        f"Genres: {genres} | "
        f"Popularity: {track['popularity']:.0f} | "
        f"Year: {track['year']} | "
        f"Duration: {track['duration_min']:.2f} min | "
        f"Explicit: {explicit} | "
        f"Spotify ID: {track['id']}"
    )


def make_example(question, selected):
    """
    Create one Qwen chat-format training example.
    """

    recommendations = []

    for i, (_, track) in enumerate(selected.iterrows(), start=1):

        recommendations.append(
            f"{i}. {format_track(track)}"
        )

    answer = "\n".join(recommendations)

    return {
        "messages": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }


def select_tracks(matching, count):
    """
    Randomly select the requested number of tracks.
    """

    if len(matching) < count:
        return None

    return matching.sample(
        n=count,
        random_state=random.randint(0, 1000000)
    )


# -----------------------------
# Available genres
# -----------------------------

genre_counts = Counter()

for _, track in spotify.iterrows():

    for genre in get_genres(track):
        genre_counts[genre] += 1


# Use the 20 most common genres that have
# enough tracks for meaningful recommendations.

genres = [
    genre
    for genre, count in genre_counts.most_common(20)
    if count >= MIN_GENRE_TRACKS
]

print("Available genres:", len(genres))
print(genres)


# -----------------------------
# Question generators
# -----------------------------

def generate_genre_example():

    if not genres:
        return None

    genre = random.choice(genres)

    count = random.choice([3, 5, 10])

    matching = spotify[
        spotify["genres"].apply(
            lambda x: genre in [
                g.strip().lower()
                for g in get_genres({"genres": x})
            ]
        )
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre} tracks.",
        f"Can you recommend {count} songs in the {genre} genre?",
        f"Give me {count} good {genre} songs.",
        f"What are {count} {genre} tracks you recommend?",
        f"I want to listen to {genre} music. Recommend {count} tracks."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_popularity_example():

    if not genres:
        return None

    genre = random.choice(genres)

    popularity = random.choice([
        50,
        60,
        70,
        80
    ])

    count = random.choice([3, 5, 10])

    matching = spotify[
        (spotify["popularity"] >= popularity) &
        spotify["genres"].apply(
            lambda x: genre in get_genres({"genres": x})
        )
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre} tracks with popularity of at least {popularity}.",
        f"Find me {count} popular {genre} songs with a Spotify popularity score of {popularity}+.",
        f"Give me {count} {genre} tracks that have a popularity score of {popularity} or higher."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_year_example():

    decade = random.choice([
        1990,
        2000,
        2010,
        2020
    ])

    count = random.choice([3, 5, 10])

    matching = spotify[
        (spotify["year"] >= decade) &
        (spotify["year"] < decade + 10)
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} songs from the {decade}s.",
        f"Give me {count} tracks released during the {decade}s.",
        f"What are {count} songs from the {decade}s that you recommend?",
        f"I want to listen to music from the {decade}s. Give me {count} recommendations."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_year_example():

    if not genres:
        return None

    genre = random.choice(genres)

    decade = random.choice([
        1990,
        2000,
        2010,
        2020
    ])

    count = random.choice([3, 5, 10])

    matching = spotify[
        (spotify["year"] >= decade) &
        (spotify["year"] < decade + 10) &
        spotify["genres"].apply(
            lambda x: genre in get_genres({"genres": x})
        )
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre} songs from the {decade}s.",
        f"Give me {count} {genre} tracks released during the {decade}s.",
        f"What are some good {genre} songs from the {decade}s?",
        f"I want {genre} music from the {decade}s. Give me {count} recommendations."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_duration_example():

    max_duration = random.choice([
        2.5,
        3.0,
        4.0,
        5.0
    ])

    count = random.choice([3, 5, 10])

    matching = spotify[
        spotify["duration_min"] < max_duration
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} songs shorter than {max_duration} minutes.",
        f"Find {count} tracks that are under {max_duration} minutes long.",
        f"Give me {count} short songs with a duration below {max_duration} minutes."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_non_explicit_example():

    count = random.choice([3, 5, 10])

    matching = spotify[
        spotify["explicit"] == False
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} non-explicit songs.",
        f"Give me {count} tracks without explicit content.",
        f"Find {count} songs that are safe to play without explicit lyrics."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_non_explicit_example():

    if not genres:
        return None

    genre = random.choice(genres)

    count = random.choice([3, 5, 10])

    matching = spotify[
        (spotify["explicit"] == False) &
        spotify["genres"].apply(
            lambda x: genre in get_genres({"genres": x})
        )
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} non-explicit {genre} tracks.",
        f"Give me {count} {genre} songs without explicit content.",
        f"Find {count} clean {genre} songs."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_artist_popularity_example():

    popularity = random.choice([
        50,
        60,
        70,
        80
    ])

    count = random.choice([3, 5, 10])

    matching = spotify[
        spotify["artist_popularity"] >= popularity
    ]

    selected = select_tracks(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} tracks by artists with popularity of at least {popularity}.",
        f"Find {count} songs from artists with a Spotify popularity score of {popularity}+.",
        f"Give me {count} tracks by popular artists with a score of {popularity} or higher."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


# -----------------------------
# Generate dataset
# -----------------------------

generators = [
    generate_genre_example,
    generate_genre_popularity_example,
    generate_year_example,
    generate_genre_year_example,
    generate_duration_example,
    generate_non_explicit_example,
    generate_genre_non_explicit_example,
    generate_artist_popularity_example
]


examples = []

attempts = 0
max_attempts = TARGET_EXAMPLES * 5

while len(examples) < TARGET_EXAMPLES and attempts < max_attempts:

    generator = random.choice(generators)

    example = generator()

    if example is not None:
        examples.append(example)

    attempts += 1


# -----------------------------
# Shuffle examples
# -----------------------------

random.shuffle(examples)


# -----------------------------
# Train / test split
# -----------------------------

test_count = int(len(examples) * TEST_SIZE)

test_examples = examples[:test_count]
train_examples = examples[test_count:]


# -----------------------------
# Save JSONL files
# -----------------------------

with open(TRAIN_FILE, "w", encoding="utf-8") as file:

    for example in train_examples:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )


with open(TEST_FILE, "w", encoding="utf-8") as file:

    for example in test_examples:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )


# -----------------------------
# Results
# -----------------------------

print()
print("Music SFT dataset created!")
print("Total examples:", len(examples))
print("Training examples:", len(train_examples))
print("Test examples:", len(test_examples))
print("Attempts:", attempts)
print()
print("Created:")
print(TRAIN_FILE)
print(TEST_FILE)
