import pandas as pd


FEATURES_PATH = "data/processed/book_features.csv"


def load_book_features():
    return pd.read_csv(
        FEATURES_PATH,
        low_memory=False
    )


def recommend_popular_books(
    books,
    top_n=10,
    min_ratings=50
):
    eligible_books = books[
        books["Explicit-Rating-Count"] >= min_ratings
    ].copy()

    # Sort stronger editions first
    eligible_books = eligible_books.sort_values(
        by=[
            "Average-Rating",
            "Explicit-Rating-Count"
        ],
        ascending=[False, False]
    )

    # Avoid recommending multiple ISBN editions
    # of the same title by the same author
    eligible_books = eligible_books.drop_duplicates(
        subset=["Book-Title", "Book-Author"],
        keep="first"
    )

    return eligible_books.head(top_n)


def main():
    books = load_book_features()

    recommendations = recommend_popular_books(
        books,
        top_n=10,
        min_ratings=50
    )

    print("Top 10 Popular Books:\n")

    print(
        recommendations[
            [
                "Book-Title",
                "Book-Author",
                "Average-Rating",
                "Explicit-Rating-Count"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
