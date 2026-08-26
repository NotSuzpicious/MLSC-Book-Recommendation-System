import pandas as pd


BOOKS_PATH = "data/raw/BX-Books.csv"
USERS_PATH = "data/raw/BX-Users.csv"
RATINGS_PATH = "data/raw/BX-Book-Ratings.csv"


def load_data():
    books = pd.read_csv(
        BOOKS_PATH,
        sep=";",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip"
    )

    users = pd.read_csv(
        USERS_PATH,
        sep=";",
        encoding="latin-1",
        na_values=["NULL"],
        low_memory=False,
        on_bad_lines="skip"
    )

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip"
    )

    return books, users, ratings

def clean_books(books):
    books = books.copy()

    # Remove rows with missing title
    books = books.dropna(subset=["Book-Title"])

    # Fill missing author and publisher
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")
    books["Publisher"] = books["Publisher"].fillna("Unknown")

    # Convert publication year to numeric
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"],
        errors="coerce"
    )

    # Keep realistic publication years
    books.loc[
        (books["Year-Of-Publication"] < 0)
        | (books["Year-Of-Publication"] > 2026),
        "Year-Of-Publication"
    ] = pd.NA

    return books

def clean_users(users):
    users = users.copy()

    # Convert age to numeric
    users["Age"] = pd.to_numeric(
        users["Age"],
        errors="coerce"
    )

    # Treat unrealistic ages as missing
    users.loc[
        (users["Age"] < 5) | (users["Age"] > 100),
        "Age"
    ] = pd.NA

    # Fill missing age with median age
    median_age = users["Age"].median()
    users["Age"] = users["Age"].fillna(median_age)

    return users

def clean_ratings(ratings, books):
    ratings = ratings.copy()

    # Keep only ratings for ISBNs that exist in the books dataset
    ratings = ratings[
        ratings["ISBN"].isin(books["ISBN"])
    ].copy()

    # Separate implicit and explicit interactions
    implicit_ratings = ratings[
        ratings["Book-Rating"] == 0
    ].copy()

    explicit_ratings = ratings[
        ratings["Book-Rating"] > 0
    ].copy()

    return ratings, implicit_ratings, explicit_ratings

def save_processed_data(
    books,
    users,
    ratings,
    implicit_ratings,
    explicit_ratings
):
    books.to_csv(
        "data/processed/books_clean.csv",
        index=False
    )

    users.to_csv(
        "data/processed/users_clean.csv",
        index=False
    )

    ratings.to_csv(
        "data/processed/ratings_clean.csv",
        index=False
    )

    implicit_ratings.to_csv(
        "data/processed/implicit_ratings.csv",
        index=False
    )

    explicit_ratings.to_csv(
        "data/processed/explicit_ratings.csv",
        index=False
    )

def main():
    books, users, ratings = load_data()

    books = clean_books(books)
    users = clean_users(users)

    ratings, implicit_ratings, explicit_ratings = clean_ratings(
        ratings,
        books
    )

    save_processed_data(
        books,
        users,
        ratings,
        implicit_ratings,
        explicit_ratings
    )

    print("Preprocessing completed successfully.")
    print(f"Books: {len(books)}")
    print(f"Users: {len(users)}")
    print(f"Valid interactions: {len(ratings)}")
    print(f"Implicit interactions: {len(implicit_ratings)}")
    print(f"Explicit ratings: {len(explicit_ratings)}")

if __name__ == "__main__":
    main()