import pandas as pd

BOOKS_PATH = "data/raw/BX-Books.csv"
USERS_PATH = "data/raw/BX-Users.csv"
RATINGS_PATH = "data/raw/BX-Book-Ratings.csv"


def inspect_dataset(name, df):
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())


def main():
    books = pd.read_csv(
        BOOKS_PATH,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    users = pd.read_csv(
        USERS_PATH,
        sep=";",
        encoding="latin-1",
        na_values=["NULL"],
        on_bad_lines="skip"
    )

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    inspect_dataset("BOOKS DATASET", books)
    inspect_dataset("USERS DATASET", users)
    inspect_dataset("RATINGS DATASET", ratings)

    print("\n" + "=" * 60)
    print("RATING DISTRIBUTION")
    print("=" * 60)

    print(ratings["Book-Rating"].value_counts().sort_index())

    print("\n" + "=" * 60)
    print("ISBN CONSISTENCY CHECK")
    print("=" * 60)

    unique_rating_isbns = ratings["ISBN"].nunique()
    unique_book_isbns = books["ISBN"].nunique()

    matched_ratings = ratings["ISBN"].isin(books["ISBN"]).sum()
    unmatched_ratings = (~ratings["ISBN"].isin(books["ISBN"])).sum()

    print(f"Unique ISBNs in books: {unique_book_isbns}")
    print(f"Unique ISBNs in ratings: {unique_rating_isbns}")
    print(f"Ratings matched to books: {matched_ratings}")
    print(f"Ratings with unknown ISBNs: {unmatched_ratings}")


if __name__ == "__main__":
    main()