import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
BOOKS_PATH = "data/processed/books_clean.csv"

RATINGS_PATH = "data/processed/explicit_ratings.csv"


def load_data():
    ratings = pd.read_csv(
        RATINGS_PATH,
        low_memory=False
    )

    books = pd.read_csv(
        BOOKS_PATH,
        low_memory=False
    )

    return ratings, books


def inspect_interactions(ratings):
    user_counts = ratings.groupby("User-ID").size()
    book_counts = ratings.groupby("ISBN").size()

    print("Collaborative Filtering Data")
    print("=" * 50)

    print(f"Explicit ratings: {len(ratings)}")
    print(f"Unique users: {ratings['User-ID'].nunique()}")
    print(f"Unique books: {ratings['ISBN'].nunique()}")

    print("\nUser activity:")
    print(f"Users with >= 5 ratings:  {(user_counts >= 5).sum()}")
    print(f"Users with >= 10 ratings: {(user_counts >= 10).sum()}")
    print(f"Users with >= 20 ratings: {(user_counts >= 20).sum()}")

    print("\nBook activity:")
    print(f"Books with >= 5 ratings:  {(book_counts >= 5).sum()}")
    print(f"Books with >= 10 ratings: {(book_counts >= 10).sum()}")
    print(f"Books with >= 20 ratings: {(book_counts >= 20).sum()}")

def filter_ratings(
    ratings,
    min_user_ratings=10,
    min_book_ratings=10
):
    user_counts = ratings.groupby("User-ID").size()

    active_users = user_counts[
        user_counts >= min_user_ratings
    ].index

    filtered = ratings[
        ratings["User-ID"].isin(active_users)
    ].copy()

    book_counts = filtered.groupby("ISBN").size()

    active_books = book_counts[
        book_counts >= min_book_ratings
    ].index

    filtered = filtered[
        filtered["ISBN"].isin(active_books)
    ].copy()

    return filtered

def create_user_item_matrix(filtered_ratings):
    matrix = filtered_ratings.pivot_table(
        index="User-ID",
        columns="ISBN",
        values="Book-Rating"
    )

    return matrix

def prepare_similarity_matrix(user_item_matrix):
    return user_item_matrix.fillna(0)

def calculate_user_similarity(similarity_matrix):
    similarity = cosine_similarity(similarity_matrix)

    user_similarity = pd.DataFrame(
        similarity,
        index=similarity_matrix.index,
        columns=similarity_matrix.index
    )

    return user_similarity

def recommend_for_user(
    user_id,
    user_item_matrix,
    user_similarity,
    top_n=10,
    neighbor_count=20,
    min_support=3
):
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in matrix.")

    target_ratings = user_item_matrix.loc[user_id]

    similar_users = (
        user_similarity.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(neighbor_count)
    )

    weighted_scores = {}
    similarity_sums = {}
    support_counts = {}

    for neighbor_id, similarity_score in similar_users.items():
        if similarity_score <= 0:
            continue

        neighbor_ratings = user_item_matrix.loc[neighbor_id]

        for isbn, rating in neighbor_ratings.dropna().items():

            # Skip books already rated by target user
            if pd.notna(target_ratings[isbn]):
                continue

            weighted_scores[isbn] = (
                weighted_scores.get(isbn, 0)
                + similarity_score * rating
            )

            similarity_sums[isbn] = (
                similarity_sums.get(isbn, 0)
                + similarity_score
            )

            support_counts[isbn] = (
                support_counts.get(isbn, 0) + 1
            )

    predictions = []

    for isbn in weighted_scores:

        if support_counts[isbn] < min_support:
            continue

        predicted_rating = (
            weighted_scores[isbn]
            / similarity_sums[isbn]
        )

        predictions.append(
            (
                isbn,
                predicted_rating,
                support_counts[isbn]
            )
        )

    predictions.sort(
        key=lambda x: (
            x[1],
            x[2]
        ),
        reverse=True
    )

    return predictions[:top_n]

def main():
    parser = argparse.ArgumentParser(
        description="User-based collaborative filtering recommender"
    )

    parser.add_argument(
        "--user",
        type=int,
        default=None,
        help="User-ID to generate recommendations for"
    )

    args = parser.parse_args()

    ratings, books = load_data()

    filtered = filter_ratings(
        ratings,
        min_user_ratings=10,
        min_book_ratings=10
    )

    matrix = create_user_item_matrix(filtered)
    similarity_matrix = prepare_similarity_matrix(matrix)

    user_similarity = calculate_user_similarity(
        similarity_matrix
    )

    if args.user is None:
        user_activity = filtered.groupby("User-ID").size()
        sample_user = user_activity.idxmax()

        print(
            f"No User-ID supplied. Using most active user: "
            f"{sample_user}"
        )
    else:
        sample_user = args.user

    if sample_user not in matrix.index:
        print(
            f"User {sample_user} is not available in the "
            "filtered collaborative-filtering dataset."
        )
        return

    recommendations = recommend_for_user(
        sample_user,
        matrix,
        user_similarity,
        top_n=10,
        neighbor_count=20,
        min_support=3
    )

    print(f"\nRecommendations for User {sample_user}")
    print("=" * 70)

    for isbn, predicted_rating, support in recommendations:
        book_info = books[
            books["ISBN"] == isbn
        ]

        if not book_info.empty:
            title = book_info.iloc[0]["Book-Title"]
            author = book_info.iloc[0]["Book-Author"]

            print(
                f"{title} | {author} | "
                f"Predicted Rating: {predicted_rating:.2f} | "
                f"Neighbor Support: {support}"
            )

if __name__ == "__main__":
    main()