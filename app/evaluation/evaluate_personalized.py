from sklearn.model_selection import train_test_split

from app.recommendation.collaborative import (
    load_data,
    filter_ratings,
    create_user_item_matrix,
    prepare_similarity_matrix,
    calculate_user_similarity,
    recommend_for_user,
)

def evaluate_hit_rate(
    test_data,
    train_matrix,
    user_similarity,
    top_n=10
):
    hits = 0
    users_evaluated = 0
    users_without_recommendations = 0

    for user_id, user_test in test_data.groupby("User-ID"):

        # User must exist in the training model
        if user_id not in train_matrix.index:
            continue

        recommendations = recommend_for_user(
            user_id,
            train_matrix,
            user_similarity,
            top_n=top_n,
            neighbor_count=20,
            min_support=3
        )

        if not recommendations:
            users_without_recommendations += 1
            continue

        recommended_isbns = {
            isbn
            for isbn, predicted_rating, support
            in recommendations
        }

        test_isbns = set(
            user_test["ISBN"]
        )

        users_evaluated += 1

        if recommended_isbns.intersection(test_isbns):
            hits += 1

    if users_evaluated == 0:
        return None

    return {
        "Hits": hits,
        "Users-Evaluated": users_evaluated,
        "Users-Without-Recommendations":
            users_without_recommendations,
        "Hit-Rate": hits / users_evaluated
    }

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

    train_matrix = create_user_item_matrix(train_data)

    similarity_matrix = prepare_similarity_matrix(
        train_matrix
    )

    user_similarity = calculate_user_similarity(
        similarity_matrix
    )

    results = evaluate_hit_rate(
        test_data,
        train_matrix,
        user_similarity,
        top_n=10
    )

    print("\nPersonalized Recommendation Evaluation")
    print("=" * 60)

    if results is None:
        print("No users could be evaluated.")
    else:
        print(
            f"Users evaluated: "
            f"{results['Users-Evaluated']}"
        )

        print(
            f"Users with at least one hit: "
            f"{results['Hits']}"
        )

        print(
            f"Users without recommendations: "
            f"{results['Users-Without-Recommendations']}"
        )

        print(
            f"Hit Rate@10: "
            f"{results['Hit-Rate']:.2%}"
        )

    print("\nTraining recommender prepared.")
    print(f"User-item matrix: {train_matrix.shape}")
    print(f"User similarity matrix: {user_similarity.shape}")

    print("Personalized evaluation dataset prepared.")
    print("=" * 60)
    print(f"Training ratings: {len(train_data)}")
    print(f"Testing ratings: {len(test_data)}")
    print(f"Test users: {test_data['User-ID'].nunique()}")


if __name__ == "__main__":
    main()
