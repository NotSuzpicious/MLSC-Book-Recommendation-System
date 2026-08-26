import argparse

from app.services.recommendation_service import (
    prepare_recommendation_system,
    get_recommendations_for_user,
)


def main():
    parser = argparse.ArgumentParser(
        description="MLSC Book Recommendation System"
    )

    parser.add_argument(
        "--user",
        type=int,
        default=242,
        help="User-ID for personalized recommendations"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to display"
    )

    args = parser.parse_args()

    print("Preparing recommendation system...")

    (
        books,
        book_features,
        filtered,
        matrix,
        user_similarity,
        tfidf_matrix
    ) = prepare_recommendation_system()

    recommendations = get_recommendations_for_user(
        args.user,
        books,
        book_features,
        filtered,
        matrix,
        user_similarity,
        tfidf_matrix,
        top_n=args.top
    )

    if not recommendations:
        print(
            f"No personalized recommendations are available "
            f"for User {args.user}."
        )
        return

    print(f"\nRecommendations for User {args.user}")
    print("=" * 80)

    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. "
            f"{item['Book-Title']} | "
            f"{item['Book-Author']} | "
            f"Score: {item['Fuzzy-Score']:.2f}"
        )


if __name__ == "__main__":
    main()