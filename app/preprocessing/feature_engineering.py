import pandas as pd


BOOKS_PATH = "data/processed/books_clean.csv"
RATINGS_PATH = "data/processed/ratings_clean.csv"
EXPLICIT_RATINGS_PATH = "data/processed/explicit_ratings.csv"


def load_processed_data():
    books = pd.read_csv(BOOKS_PATH, low_memory=False)
    ratings = pd.read_csv(RATINGS_PATH, low_memory=False)
    explicit_ratings = pd.read_csv(
        EXPLICIT_RATINGS_PATH,
        low_memory=False
    )

    return books, ratings, explicit_ratings

def create_book_features(books, ratings, explicit_ratings):
    interaction_counts = (
        ratings.groupby("ISBN")
        .size()
        .reset_index(name="Interaction-Count")
    )

    explicit_stats = (
        explicit_ratings.groupby("ISBN")["Book-Rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "Explicit-Rating-Count",
                "mean": "Average-Rating"
            }
        )
    )

    book_features = books.merge(
        interaction_counts,
        on="ISBN",
        how="left"
    )

    book_features = book_features.merge(
        explicit_stats,
        on="ISBN",
        how="left"
    )

    book_features["Interaction-Count"] = (
        book_features["Interaction-Count"]
        .fillna(0)
        .astype(int)
    )

    book_features["Explicit-Rating-Count"] = (
        book_features["Explicit-Rating-Count"]
        .fillna(0)
        .astype(int)
    )

    book_features["Average-Rating"] = (
        book_features["Average-Rating"]
        .fillna(0)
        .round(2)
    )

    return book_features

def save_book_features(book_features):
    book_features.to_csv(
        "data/processed/book_features.csv",
        index=False
    )

def main():
    books, ratings, explicit_ratings = load_processed_data()

    book_features = create_book_features(
        books,
        ratings,
        explicit_ratings
    )

    save_book_features(book_features)

    print("Book features created and saved successfully.")
    print(f"Shape: {book_features.shape}")

if __name__ == "__main__":
    main()