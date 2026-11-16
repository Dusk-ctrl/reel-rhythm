import pandas as pd

# -----------------------------------------
# 1. Load the Spotify dataset
# -----------------------------------------

spotify = pd.read_csv(
    "spotify_data.csv",
    na_values=["N/A", ""],
    low_memory=False
)

print("Original dataset size:", len(spotify))

# -----------------------------------------
# 2. Keep only the columns we need
# -----------------------------------------

spotify = spotify[[
    "track_id",
    "track_name",
    "artist_name",
    "track_popularity",
    "artist_popularity",
    "artist_followers",
    "artist_genres",
    "album_name",
    "album_release_date",
    "track_duration_min",
    "explicit"
]].copy()

# Rename columns
spotify = spotify.rename(columns={
    "track_id": "id",
    "track_name": "title",
    "artist_name": "artist",
    "track_popularity": "popularity",
    "artist_genres": "genres",
    "album_name": "album",
    "album_release_date": "release_date",
    "track_duration_min": "duration_min"
})

# Remove rows missing essential information
spotify = spotify.dropna(subset=[
    "id",
    "title",
    "artist",
    "popularity"
])

# Remove extra whitespace from text columns
text_columns = [
    "id",
    "title",
    "artist",
    "genres",
    "album"
]

for column in text_columns:
    spotify[column] = spotify[column].astype("string").str.strip()

# Convert numeric columns
numeric_columns = [
    "popularity",
    "artist_popularity",
    "artist_followers",
    "duration_min"
]

for column in numeric_columns:
    spotify[column] = pd.to_numeric(
        spotify[column],
        errors="coerce"
    )

# Convert explicit to boolean
spotify["explicit"] = (
    spotify["explicit"]
    .astype("string")
    .str.upper()
    .map({"TRUE": True, "FALSE": False})
)

# Convert release date to datetime
spotify["release_date"] = pd.to_datetime(
    spotify["release_date"],
    errors="coerce"
)

# -----------------------------------------
# 3. Remove invalid values
# -----------------------------------------

# Remove invalid popularity values

spotify = spotify[
    (spotify["popularity"] >= 0) &
    (spotify["popularity"] <= 100)
]

spotify = spotify[
    (spotify["artist_popularity"].isna()) |
    ((spotify["artist_popularity"] >= 0) &
     (spotify["artist_popularity"] <= 100))
]

# Remove tracks with invalid durations
spotify = spotify[
    (spotify["duration_min"] > 0) &
    (spotify["duration_min"] < 30)
]

# Remove duplicate tracks
spotify = spotify.drop_duplicates(subset=["id"])

# Reset row numbering
spotify = spotify.reset_index(drop=True)

# Save cleaned dataset
spotify.to_csv("spotify_clean.csv", index=False)

# -----------------------------------------
# 4. Display results
# -----------------------------------------

print("\nCleaning complete!")
print("Cleaned dataset size:", len(spotify))
print("\nMissing values:")
print(spotify.isna().sum())

print("\nFirst five rows:")
print(spotify.head())

print("\nSaved as spotify_clean.csv")
