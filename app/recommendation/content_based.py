import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOOKS_PATH = "data/processed/books_clean.csv"


def load_books():
    return pd.read_csv(
        BOOKS_PATH,
        low_memory=False
    )


def inspect_metadata(books):
    print("Content-Based Recommendation Metadata")
    print("=" * 55)

    print(f"Total books: {len(books)}")

    columns = [
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication"
    ]

    print("\nMissing values:")
    print(books[columns].isnull().sum())

    print("\nUnique values:")
    print(f"Titles: {books['Book-Title'].nunique()}")
    print(f"Authors: {books['Book-Author'].nunique()}")
    print(f"Publishers: {books['Publisher'].nunique()}")

    print("\nSample metadata:")
    print(
        books[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Publisher",
                "Year-Of-Publication"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

def build_content_features(books):
    books = books.copy()

    books["Content-Text"] = (
        books["Book-Title"].fillna("")
        + " "
        + books["Book-Author"].fillna("")
        + " "
        + books["Publisher"].fillna("")
    )

    return books

def create_tfidf_matrix(books):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    tfidf_matrix = vectorizer.fit_transform(
        books["Content-Text"]
    )

    return vectorizer, tfidf_matrix

def recommend_similar_books(
    title,
    books,
    tfidf_matrix,
    top_n=10
):
    matches = books[
        books["Book-Title"].str.lower() == title.lower()
    ]

    if matches.empty:
        raise ValueError(f"Book title not found: {title}")

    book_index = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    recommendations = []
    seen_books = set()

    original_title = str(
        books.iloc[book_index]["Book-Title"]
    ).strip().lower()

    original_author = str(
        books.iloc[book_index]["Book-Author"]
    ).strip().lower()

    seen_books.add(
        (original_title, original_author)
    )

    for index in similar_indices:
        candidate_title = str(
            books.iloc[index]["Book-Title"]
        ).strip()

        candidate_author = str(
            books.iloc[index]["Book-Author"]
        ).strip()

        book_key = (
            candidate_title.lower(),
            candidate_author.lower()
        )

        if book_key in seen_books:
            continue

        seen_books.add(book_key)

        recommendations.append(
            (
                candidate_title,
                candidate_author,
                similarity_scores[index]
            )
        )

        if len(recommendations) == top_n:
            break

    return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Content-based book recommender"
    )

    parser.add_argument(
        "--title",
        type=str,
        default="The Da Vinci Code",
        help="Book title to find similar books for"
    )

    args = parser.parse_args()

    books = load_books()
    books = build_content_features(books)

    vectorizer, tfidf_matrix = create_tfidf_matrix(books)

    recommendations = recommend_similar_books(
        args.title,
        books,
        tfidf_matrix,
        top_n=10
    )

    print(f"Books similar to: {args.title}")
    print("=" * 70)

    for title, author, score in recommendations:
        print(
            f"{title} | {author} | "
            f"Similarity: {score:.3f}"
        )


if __name__ == "__main__":
    main()