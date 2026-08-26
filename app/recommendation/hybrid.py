import pandas as pd
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from app.recommendation.collaborative import (
    filter_ratings,
    create_user_item_matrix,
    prepare_similarity_matrix,
    calculate_user_similarity,
    recommend_for_user,
)

from app.fuzzy.fuzzy_recommender import (
    create_fuzzy_system,
    evaluate_fuzzy_recommendation,
)

from app.recommendation.content_based import (
    build_content_features,
    create_tfidf_matrix,
)

BOOKS_PATH = "data/processed/books_clean.csv"
RATINGS_PATH = "data/processed/explicit_ratings.csv"
FEATURES_PATH = "data/processed/book_features.csv"


def load_data():
    books = pd.read_csv(
        BOOKS_PATH,
        low_memory=False
    )

    ratings = pd.read_csv(
        RATINGS_PATH,
        low_memory=False
    )

    book_features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False
    )

    return books, ratings, book_features

def get_collaborative_candidates(
    user_id,
    matrix,
    user_similarity,
    top_n=30
):
    return recommend_for_user(
        user_id,
        matrix,
        user_similarity,
        top_n=top_n,
        neighbor_count=20,
        min_support=3
    )

def build_user_content_profile(
    user_id,
    filtered_ratings,
    books,
    tfidf_matrix,
    min_rating=7
):
    user_ratings = filtered_ratings[
        filtered_ratings["User-ID"] == user_id
    ]

    liked_books = user_ratings[
        user_ratings["Book-Rating"] >= min_rating
    ]

    liked_indices = books[
        books["ISBN"].isin(liked_books["ISBN"])
    ].index.tolist()

    if not liked_indices:
        return None

    profile = tfidf_matrix[liked_indices].mean(axis=0)

    return np.asarray(profile)

def score_candidates_by_content(
    candidates,
    books,
    tfidf_matrix,
    user_profile
):
    scored_candidates = []

    isbn_to_index = {
        isbn: index
        for index, isbn in books["ISBN"].items()
    }

    for isbn, predicted_rating, support in candidates:
        if isbn not in isbn_to_index:
            continue

        book_index = isbn_to_index[isbn]

        content_score = cosine_similarity(
            user_profile,
            tfidf_matrix[book_index]
        )[0][0]

        scored_candidates.append(
            (
                isbn,
                predicted_rating,
                support,
                content_score
            )
        )

    return scored_candidates

def calculate_hybrid_scores(
    scored_candidates,
    collaborative_weight=0.7,
    content_weight=0.3
):
    hybrid_results = []

    for isbn, predicted_rating, support, content_score in scored_candidates:

        collaborative_score = predicted_rating / 10.0

        hybrid_score = (
            collaborative_weight * collaborative_score
            + content_weight * content_score
        )

        hybrid_results.append(
            (
                isbn,
                predicted_rating,
                content_score,
                hybrid_score,
                support
            )
        )

    hybrid_results.sort(
        key=lambda x: x[3],
        reverse=True
    )

    return hybrid_results

def normalize_value(value, maximum):
    if maximum <= 0:
        return 0.0

    normalized = (
        np.log1p(value)
        / np.log1p(maximum)
    )

    return min(float(normalized), 1.0)

def main():

    final_results = []

    parser = argparse.ArgumentParser(
        description="Hybrid book recommender"
    )

    parser.add_argument(
        "--user",
        type=int,
        default=242,
        help="User-ID to generate hybrid recommendations for"
    )

    args = parser.parse_args()

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
    vectorizer, tfidf_matrix = create_tfidf_matrix(books)

    sample_user = args.user

    if sample_user not in matrix.index:
        print(
            f"User {sample_user} is not available in the "
            "filtered hybrid-recommendation dataset."
        )
        return

    candidates = get_collaborative_candidates(
        sample_user,
        matrix,
        user_similarity,
        top_n=30
    )

    user_profile = build_user_content_profile(
        sample_user,
        filtered,
        books,
        tfidf_matrix,
        min_rating=7
    )

    if user_profile is None:
        print(
            f"No suitable content profile could be created "
            f"for User {sample_user}."
        )
        return

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

    max_interactions = book_features[
        "Interaction-Count"
    ].max()

    max_rating_count = book_features[
        "Explicit-Rating-Count"
    ].max()

    (
        personal_relevance_var,
        popularity_var,
        rating_confidence_var,
        recommendation_score_var,
        fuzzy_simulation
    ) = create_fuzzy_system()

    
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

        final_results.append(
            (
                isbn,
                hybrid_score,
                popularity_score,
                confidence_score,
                fuzzy_score
            )
        )

    final_results.sort(
        key=lambda x: x[4],
        reverse=True
    )

    print(f"\nFinal Fuzzy Recommendations for User {sample_user}")
    print("=" * 90)

    for (
        isbn,
        hybrid_score,
        popularity_score,
        confidence_score,
        fuzzy_score
    ) in final_results[:10]:

        book_info = books[
            books["ISBN"] == isbn
        ]

        if book_info.empty:
            continue

        title = book_info.iloc[0]["Book-Title"]
        author = book_info.iloc[0]["Book-Author"]

        print(
            f"{title} | {author} | "
            f"Hybrid: {hybrid_score:.3f} | "
            f"Popularity: {popularity_score:.3f} | "
            f"Confidence: {confidence_score:.3f} | "
            f"Fuzzy Score: {fuzzy_score:.2f}"
        )


if __name__ == "__main__":
    main()