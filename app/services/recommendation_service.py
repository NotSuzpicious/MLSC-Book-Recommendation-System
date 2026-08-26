from app.recommendation.hybrid import (
    load_data,
    filter_ratings,
    create_user_item_matrix,
    prepare_similarity_matrix,
    calculate_user_similarity,
    build_content_features,
    create_tfidf_matrix,
    get_collaborative_candidates,
    build_user_content_profile,
    score_candidates_by_content,
    calculate_hybrid_scores,
)

from app.fuzzy.fuzzy_recommender import (
    create_fuzzy_system,
    evaluate_fuzzy_recommendation,
)

import numpy as np

def normalize_value(value, maximum):
    if maximum <= 0:
        return 0.0

    return min(
        float(np.log1p(value) / np.log1p(maximum)),
        1.0
    )

def prepare_recommendation_system():
    books, ratings, book_features = load_data()

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

    books = build_content_features(books)
    _, tfidf_matrix = create_tfidf_matrix(books)

    return (
        books,
        book_features,
        filtered,
        matrix,
        user_similarity,
        tfidf_matrix
    )

def get_recommendations_for_user(
    user_id,
    books,
    book_features,
    filtered,
    matrix,
    user_similarity,
    tfidf_matrix,
    top_n=10
):
    if user_id not in matrix.index:
        return []

    candidates = get_collaborative_candidates(
        user_id,
        matrix,
        user_similarity,
        top_n=30
    )

    user_profile = build_user_content_profile(
        user_id,
        filtered,
        books,
        tfidf_matrix,
        min_rating=7
    )

    if user_profile is None:
        return []

    scored_candidates = score_candidates_by_content(
        candidates,
        books,
        tfidf_matrix,
        user_profile
    )

    hybrid_results = calculate_hybrid_scores(
        scored_candidates,
        collaborative_weight=0.7,
        content_weight=0.3
    )

    max_interactions = book_features["Interaction-Count"].max()
    max_rating_count = book_features["Explicit-Rating-Count"].max()

    (
        _,
        _,
        _,
        _,
        fuzzy_simulation
    ) = create_fuzzy_system()

    final_results = []

    for (
        isbn,
        predicted_rating,
        content_score,
        hybrid_score,
        support
    ) in hybrid_results:

        feature_info = book_features[
            book_features["ISBN"] == isbn
        ]

        if feature_info.empty:
            continue

        interaction_count = feature_info.iloc[0]["Interaction-Count"]
        explicit_rating_count = feature_info.iloc[0]["Explicit-Rating-Count"]

        popularity_score = normalize_value(
            interaction_count,
            max_interactions
        )

        confidence_score = normalize_value(
            explicit_rating_count,
            max_rating_count
        )

        fuzzy_score = evaluate_fuzzy_recommendation(
            fuzzy_simulation,
            personal_relevance=hybrid_score,
            popularity=popularity_score,
            rating_confidence=confidence_score
        )

        book_info = books[
            books["ISBN"] == isbn
        ]

        if book_info.empty:
            continue

        final_results.append(
            {
                "ISBN": isbn,
                "Book-Title": book_info.iloc[0]["Book-Title"],
                "Book-Author": book_info.iloc[0]["Book-Author"],
                "Hybrid-Score": hybrid_score,
                "Popularity-Score": popularity_score,
                "Confidence-Score": confidence_score,
                "Fuzzy-Score": fuzzy_score,
            }
        )

    final_results.sort(
        key=lambda item: item["Fuzzy-Score"],
        reverse=True
    )

    return final_results[:top_n]

def main():
    (
        books,
        book_features,
        filtered,
        matrix,
        user_similarity,
        tfidf_matrix
    ) = prepare_recommendation_system()

    recommendations = get_recommendations_for_user(
        242,
        books,
        book_features,
        filtered,
        matrix,
        user_similarity,
        tfidf_matrix,
        top_n=5
    )

    print("\nService Recommendations for User 242")
    print("=" * 70)

    for item in recommendations:
        print(
            f"{item['Book-Title']} | "
            f"{item['Book-Author']} | "
            f"Fuzzy Score: {item['Fuzzy-Score']:.2f}"
        )

if __name__ == "__main__":
    main()