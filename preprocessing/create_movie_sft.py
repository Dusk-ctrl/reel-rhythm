import pandas as pd
import json
import random


# -----------------------------
# Settings
# -----------------------------

INPUT_FILE = "movies_clean.csv"
TRAIN_FILE = "movie_train.jsonl"
TEST_FILE = "movie_test.jsonl"

RANDOM_SEED = 42
TEST_SIZE = 0.20

TARGET_EXAMPLES = 1500

MIN_RATING = 7.0
MIN_VOTES = 1000
MAX_YEAR = 2025

random.seed(RANDOM_SEED)


# -----------------------------
# Load dataset
# -----------------------------

movies = pd.read_csv(INPUT_FILE)

print("Movies loaded:", len(movies))


# -----------------------------
# Basic cleanup
# -----------------------------

movies["year"] = pd.to_numeric(
    movies["year"],
    errors="coerce"
)

movies["rating"] = pd.to_numeric(
    movies["rating"],
    errors="coerce"
)

movies["votes"] = pd.to_numeric(
    movies["votes"],
    errors="coerce"
)

movies["genres"] = movies["genres"].fillna("")


# -----------------------------
# Filter movies
# -----------------------------

movies = movies.dropna(
    subset=[
        "title",
        "year",
        "rating",
        "votes"
    ]
)

movies = movies[
    (movies["rating"] >= MIN_RATING) &
    (movies["votes"] >= MIN_VOTES) &
    (movies["year"] <= MAX_YEAR)
].copy()

movies["year"] = movies["year"].astype(int)

print("Movies after filtering:", len(movies))


# -----------------------------
# Helper functions
# -----------------------------

def get_genres(movie):
    """
    Convert the genre string into a list.

    Example:
    "Action,Crime,Drama"

    becomes:
    ["action", "crime", "drama"]
    """

    return [
        genre.strip().lower()
        for genre in str(movie["genres"]).split(",")
        if genre.strip()
    ]


def has_genre(movie, genre):
    """
    Check whether a movie belongs to a genre.
    """

    return genre.lower() in get_genres(movie)


def format_movie(movie):
    """
    Format one movie for the assistant's response.
    """

    return (
        f"{movie['title']} "
        f"({movie['year']}) - "
        f"{movie['genres']} - "
        f"IMDb: {movie['rating']:.1f}"
    )


def make_example(question, selected):
    """
    Create one Qwen chat-format training example.
    """

    recommendations = []

    for i, (_, movie) in enumerate(selected.iterrows(), start=1):
        recommendations.append(
            f"{i}. {format_movie(movie)}"
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


def select_movies(matching, count):
    """
    Randomly select the requested number of movies.
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

genres = [
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
# Question generators
# -----------------------------

def generate_genre_example():

    genre = random.choice(genres)

    count = random.choice([3, 5, 10])

    matching = movies[
        movies["genres"].apply(
            lambda x: genre.lower() in [
                g.strip().lower()
                for g in str(x).split(",")
            ]
        )
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre.lower()} movies.",
        f"Can you recommend {count} {genre.lower()} movies?",
        f"Give me {count} good {genre.lower()} movies.",
        f"What are {count} {genre.lower()} movies you recommend?",
        f"I want to watch {count} {genre.lower()} movies. What should I watch?"
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_rating_example():

    genre = random.choice(genres)

    rating = random.choice([
        7.0,
        7.5,
        8.0,
        8.5
    ])

    count = random.choice([3, 5, 10])

    matching = movies[
        (movies["rating"] >= rating) &
        movies["genres"].apply(
            lambda x: genre.lower() in [
                g.strip().lower()
                for g in str(x).split(",")
            ]
        )
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre.lower()} movies rated at least {rating}.",
        f"Give me {count} {genre.lower()} movies with an IMDb rating of {rating} or higher.",
        f"What are {count} highly rated {genre.lower()} movies?",
        f"Find me {count} {genre.lower()} movies rated {rating}+."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_decade_example():

    decade = random.choice([
        1970,
        1980,
        1990,
        2000,
        2010,
        2020
    ])

    count = random.choice([3, 5, 10])

    matching = movies[
        (movies["year"] >= decade) &
        (movies["year"] < decade + 10)
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} movies from the {decade}s.",
        f"Give me {count} good movies from the {decade}s.",
        f"What are {count} movies from the {decade}s that you recommend?",
        f"I want to watch movies from the {decade}s. Give me {count} recommendations."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_decade_example():

    genre = random.choice(genres)

    decade = random.choice([
        1970,
        1980,
        1990,
        2000,
        2010,
        2020
    ])

    count = random.choice([3, 5, 10])

    matching = movies[
        (movies["year"] >= decade) &
        (movies["year"] < decade + 10) &
        movies["genres"].apply(
            lambda x: genre.lower() in [
                g.strip().lower()
                for g in str(x).split(",")
            ]
        )
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre.lower()} movies from the {decade}s.",
        f"Give me {count} {genre.lower()} movies from the {decade}s.",
        f"What are some good {genre.lower()} movies from the {decade}s?",
        f"I want {genre.lower()} movies from the {decade}s. Give me {count} recommendations."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_genre_decade_rating_example():

    genre = random.choice(genres)

    decade = random.choice([
        1970,
        1980,
        1990,
        2000,
        2010,
        2020
    ])

    rating = random.choice([
        7.0,
        7.5,
        8.0
    ])

    count = random.choice([3, 5, 10])

    matching = movies[
        (movies["year"] >= decade) &
        (movies["year"] < decade + 10) &
        (movies["rating"] >= rating) &
        movies["genres"].apply(
            lambda x: genre.lower() in [
                g.strip().lower()
                for g in str(x).split(",")
            ]
        )
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} {genre.lower()} movies from the {decade}s rated at least {rating}.",
        f"Give me {count} highly rated {genre.lower()} movies from the {decade}s.",
        f"Find {count} {genre.lower()} movies from the {decade}s with an IMDb rating of {rating}+."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_year_range_example():

    start_year = random.choice([
        1980,
        1990,
        2000,
        2010
    ])

    end_year = start_year + 9

    count = random.choice([3, 5, 10])

    matching = movies[
        (movies["year"] >= start_year) &
        (movies["year"] <= end_year)
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} movies released between {start_year} and {end_year}.",
        f"Give me {count} good movies from {start_year} to {end_year}.",
        f"What are {count} movies from {start_year}-{end_year} that you recommend?"
    ]

    question = random.choice(questions)

    return make_example(question, selected)


def generate_rating_example():

    rating = random.choice([
        7.0,
        7.5,
        8.0,
        8.5
    ])

    count = random.choice([3, 5, 10])

    matching = movies[
        movies["rating"] >= rating
    ]

    selected = select_movies(matching, count)

    if selected is None:
        return None

    questions = [
        f"Recommend {count} movies rated at least {rating}.",
        f"Give me {count} highly rated movies with an IMDb rating of {rating}+.",
        f"What are {count} movies rated {rating} or higher?",
        f"Find me {count} movies with an IMDb rating of at least {rating}."
    ]

    question = random.choice(questions)

    return make_example(question, selected)


# -----------------------------
# Generate dataset
# -----------------------------

generators = [
    generate_genre_example,
    generate_genre_rating_example,
    generate_decade_example,
    generate_genre_decade_example,
    generate_genre_decade_rating_example,
    generate_year_range_example,
    generate_rating_example
]


examples = []

attempts = 0
max_attempts = TARGET_EXAMPLES * 3

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
print("SFT dataset created!")
print("Total examples:", len(examples))
print("Training examples:", len(train_examples))
print("Test examples:", len(test_examples))
print()
print("Created:")
print(TRAIN_FILE)
print(TEST_FILE)
