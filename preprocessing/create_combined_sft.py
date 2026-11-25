import pandas as pd
import json
import random
import ast

# -----------------------------
# SETTINGS
# -----------------------------

MOVIE_FILE = "movies_clean.csv"
MUSIC_FILE = "spotify_clean.csv"

TRAIN_FILE = "combined_train.jsonl"
TEST_FILE = "combined_test.jsonl"

TARGET_EXAMPLES = 400
TEST_SIZE = 0.20

random.seed(42)


# -----------------------------
# GENRE FUNCTIONS
# -----------------------------

def get_movie_genres(value):
    if pd.isna(value):
        return []

    return [
        genre.strip()
        for genre in str(value).split(",")
        if genre.strip()
    ]


def get_music_genres(value):
    if pd.isna(value):
        return []

    # Handle genres stored as Python lists
    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return [
                str(genre).strip().lower()
                for genre in parsed
                if str(genre).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    # Handle comma-separated genres
    return [
        genre.strip().lower()
        for genre in str(value).split(",")
        if genre.strip()
    ]


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

movies["year"] = movies["year"].astype(int)

# Use the same quality filters as the movie dataset
movies = movies[
    (movies["rating"] >= 7.0) &
    (movies["votes"] >= 1000) &
    (movies["year"] <= 2025)
].copy()

movies["genre_list"] = movies["genres"].apply(get_movie_genres)


# -----------------------------
# LOAD MUSIC
# -----------------------------

music = pd.read_csv(MUSIC_FILE)

music["popularity"] = pd.to_numeric(
    music["popularity"], errors="coerce"
)

music["artist_popularity"] = pd.to_numeric(
    music["artist_popularity"], errors="coerce"
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

music["year"] = music["year"].astype(int)

music = music[
    (music["popularity"] >= 0) &
    (music["popularity"] <= 100) &
    (music["year"] <= 2025) &
    (music["duration_min"] > 0) &
    (music["duration_min"] < 30)
].copy()

music["genre_list"] = music["genres"].apply(get_music_genres)


# -----------------------------
# FIND AVAILABLE GENRES
# -----------------------------

movie_genres = []

for genre_list in movies["genre_list"]:
    for genre in genre_list:
        if genre not in movie_genres:
            movie_genres.append(genre)

# Only use music genres with at least 10 tracks
music_genre_counts = {}

for genre_list in music["genre_list"]:
    for genre in set(genre_list):
        if genre not in music_genre_counts:
            music_genre_counts[genre] = 0

        music_genre_counts[genre] += 1

music_genres = [
    genre
    for genre, count in music_genre_counts.items()
    if count >= 10
]

print("Movies loaded:", len(movies))
print("Music tracks loaded:", len(music))
print("Available movie genres:", len(movie_genres))
print("Available music genres:", len(music_genres))


# -----------------------------
# FORMAT RECOMMENDATIONS
# -----------------------------

def format_movie(movie, number):
    return (
        f"{number}. {movie['title']} ({movie['year']})"
        f" | Genres: {movie['genres']}"
        f" | Rating: {movie['rating']:.1f}"
        f" | Votes: {int(movie['votes'])}"
        f" | IMDb ID: {movie['id']}"
    )


def format_track(track, number):
    explicit = str(track["explicit"]).strip().lower()

    if explicit in ["true", "1", "yes"]:
        explicit_text = "Yes"
    else:
        explicit_text = "No"

    return (
        f"{number}. {track['title']} - {track['artist']}"
        f" | Genres: {track['genres']}"
        f" | Popularity: {int(track['popularity'])}"
        f" | Year: {track['year']}"
        f" | Duration: {track['duration_min']:.2f} min"
        f" | Explicit: {explicit_text}"
        f" | Spotify ID: {track['id']}"
    )


# -----------------------------
# CREATE ONE TRAINING EXAMPLE
# -----------------------------

def make_example():
    movie_genre = random.choice(movie_genres)
    music_genre = random.choice(music_genres)

    movie_count = random.choice([3, 5])
    music_count = random.choice([3, 5])

    # Find movies matching the selected genre
    movie_pool = movies[
        movies["genre_list"].apply(
            lambda genres: movie_genre in genres
        )
    ]

    # Find tracks matching the selected genre
    music_pool = music[
        music["genre_list"].apply(
            lambda genres: music_genre in genres
        )
    ]

    # Make sure enough recommendations exist
    if len(movie_pool) < movie_count:
        return None

    if len(music_pool) < music_count:
        return None

    selected_movies = movie_pool.sample(
        n=movie_count
    )

    selected_tracks = music_pool.sample(
        n=music_count
    )

    # Build the prompt
    question = (
        f"Recommend {movie_count} {movie_genre} movies "
        f"and {music_count} {music_genre} tracks."
    )

    # Build the response
    response = "Movies:\n"

    for i, (_, movie) in enumerate(
        selected_movies.iterrows(), start=1
    ):
        response += format_movie(movie, i) + "\n"

    response += "\nMusic:\n"

    for i, (_, track) in enumerate(
        selected_tracks.iterrows(), start=1
    ):
        response += format_track(track, i) + "\n"

    return {
        "messages": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": response.strip()
            }
        ]
    }


# -----------------------------
# GENERATE DATASET
# -----------------------------

examples = []

attempts = 0
max_attempts = TARGET_EXAMPLES * 10

while len(examples) < TARGET_EXAMPLES:
    attempts += 1

    if attempts > max_attempts:
        break

    example = make_example()

    if example is not None:
        examples.append(example)

random.shuffle(examples)

split_index = int(
    len(examples) * (1 - TEST_SIZE)
)

train_examples = examples[:split_index]
test_examples = examples[split_index:]


# -----------------------------
# SAVE JSONL FILES
# -----------------------------

def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        for example in data:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


save_jsonl(TRAIN_FILE, train_examples)
save_jsonl(TEST_FILE, test_examples)


# -----------------------------
# RESULTS
# -----------------------------

print("\nCombined SFT dataset created!")

print("Total examples:", len(examples))
print("Training examples:", len(train_examples))
print("Test examples:", len(test_examples))
print("Attempts:", attempts)

print("\nCreated:")
print(TRAIN_FILE)
print(TEST_FILE)

if len(examples) < TARGET_EXAMPLES:
    print("\nWARNING: Could not generate all requested examples.")
