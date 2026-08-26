from sklearn.model_selection import train_test_split

from app.recommendation.collaborative import (
    load_data,
    filter_ratings,
)

from app.recommendation.popularity import (
    load_book_features,
    recommend_popular_books,
)


def main():
    ratings, _ = load_data()

    filtered = filter_ratings(
        ratings,
        min_user_ratings=10,
        min_book_ratings=10
    )

    train_data, test_data = train_test_split(
        filtered,
        test_size=0.2,
        random_state=42
    )

    book_features = load_book_features()

    popular_books = recommend_popular_books(
        book_features,
        top_n=10,
        min_ratings=50
    )

    popular_isbns = set(
        popular_books["ISBN"]
    )

    hits = 0
    total_users = 0

    for user_id, user_test in test_data.groupby("User-ID"):
        test_isbns = set(
            user_test["ISBN"]
        )

        total_users += 1

        if popular_isbns.intersection(test_isbns):
            hits += 1

    hit_rate = hits / total_users

    print("Popularity Baseline Evaluation")
    print("=" * 60)
    print(f"Users evaluated: {total_users}")
    print(f"Users with at least one hit: {hits}")
    print(f"Hit Rate@10: {hit_rate:.2%}")


if __name__ == "__main__":
    main()