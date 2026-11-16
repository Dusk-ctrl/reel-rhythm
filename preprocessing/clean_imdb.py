import pandas as pd
import os

# -----------------------------------------
# 1. Load the ratings dataset
# -----------------------------------------

print("Loading ratings...")

ratings = pd.read_csv(
    "title.ratings.tsv",
    sep="\t",
    usecols=["tconst", "averageRating", "numVotes"],
    dtype={"tconst": "string"},
    na_values=["\\N"]
)

print("Ratings loaded:", len(ratings))


# -----------------------------------------
# 2. Process the basics dataset in chunks
# -----------------------------------------

print("Cleaning movie data...")

output_file = "movies_clean.csv"

# Remove an old output file so we start fresh
if os.path.exists(output_file):
    os.remove(output_file)

chunk_size = 500000
first_chunk = True
total_movies = 0

for chunk in pd.read_csv(
    "title.basics.tsv",
    sep="\t",
    usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
    dtype={"tconst": "string"},
    na_values=["\\N"],
    chunksize=chunk_size
):

    # Keep feature-length movies only
    chunk = chunk[chunk["titleType"] == "movie"].copy()

    # Keep only the columns we need
    chunk = chunk[["tconst", "primaryTitle", "startYear", "genres"]]

    # Match each movie with its rating information
    movies = chunk.merge(
        ratings,
        on="tconst",
        how="inner"
    )

    # Rename columns
    movies = movies.rename(columns={
        "tconst": "id",
        "primaryTitle": "title",
        "startYear": "year",
        "averageRating": "rating",
        "numVotes": "votes"
    })

    # Convert numeric columns
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["rating"] = pd.to_numeric(movies["rating"], errors="coerce")
    movies["votes"] = pd.to_numeric(movies["votes"], errors="coerce")

    # Remove rows with missing required information
    movies = movies.dropna(
        subset=["id", "title", "year", "genres", "rating", "votes"]
    )

    # Keep valid rating values
    movies = movies[
        (movies["rating"] >= 0) &
        (movies["rating"] <= 10) &
        (movies["votes"] >= 0)
    ]

    # Convert year to integer
    movies["year"] = movies["year"].astype(int)

    # Remove duplicate movie IDs
    movies = movies.drop_duplicates(subset=["id"])

    # Arrange columns in the desired order
    movies = movies[
        ["id", "title", "year", "genres", "rating", "votes"]
    ]

    # Add this chunk to the output CSV
    movies.to_csv(
        output_file,
        index=False,
        mode="w" if first_chunk else "a",
        header=first_chunk
    )

    total_movies += len(movies)
    first_chunk = False

    print("Processed movies:", total_movies)


# -----------------------------------------
# 3. Final results
# -----------------------------------------

print("\nCleaning complete!")
print("Total movies saved:", total_movies)
print("Output file:", output_file)
