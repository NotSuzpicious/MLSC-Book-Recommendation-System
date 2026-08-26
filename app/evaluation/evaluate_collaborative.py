import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from app.recommendation.collaborative import (
    load_data,
    filter_ratings,
    create_user_item_matrix,
    prepare_similarity_matrix,
    calculate_user_similarity,
)

def evaluate_model(
    test_data,
    train_matrix,
    user_similarity
):
    actual_ratings = []
    predicted_ratings = []

    for _, row in test_data.iterrows():
        prediction = predict_rating(
            row["User-ID"],
            row["ISBN"],
            train_matrix,
            user_similarity
        )

        if prediction is None:
            continue

        actual_ratings.append(
            row["Book-Rating"]
        )

        predicted_ratings.append(
            prediction
        )

    if not predicted_ratings:
        return None

    mae = mean_absolute_error(
        actual_ratings,
        predicted_ratings
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_ratings,
            predicted_ratings
        )
    )

    coverage = (
        len(predicted_ratings)
        / len(test_data)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Coverage": coverage,
        "Predictions": len(predicted_ratings),
        "Test-Ratings": len(test_data),
    }

def split_data(filtered):
    train_data, test_data = train_test_split(
        filtered,
        test_size=0.2,
        random_state=42
    )

    return train_data, test_data

def predict_rating(
    user_id,
    isbn,
    train_matrix,
    user_similarity,
    neighbor_count=20
):
    if user_id not in train_matrix.index:
        return None

    if isbn not in train_matrix.columns:
        return None

    similar_users = (
        user_similarity.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(neighbor_count)
    )

    weighted_sum = 0.0
    similarity_sum = 0.0

    for neighbor_id, similarity_score in similar_users.items():
        if similarity_score <= 0:
            continue

        neighbor_rating = train_matrix.loc[
            neighbor_id,
            isbn
        ]

        if pd.isna(neighbor_rating):
            continue

        weighted_sum += (
            similarity_score * neighbor_rating
        )

        similarity_sum += similarity_score

    if similarity_sum == 0:
        return None

    return weighted_sum / similarity_sum

def main():
    ratings, _ = load_data()

    filtered = filter_ratings(
        ratings,
        min_user_ratings=10,
        min_book_ratings=10
    )

    train_data, test_data = split_data(filtered)

    train_matrix = create_user_item_matrix(train_data)

    train_similarity_matrix = prepare_similarity_matrix(
        train_matrix
    )

    user_similarity = calculate_user_similarity(
        train_similarity_matrix
    )

    print("Training model prepared.")
    print("=" * 60)
    print(f"User-item matrix shape: {train_matrix.shape}")
    print(f"User similarity shape: {user_similarity.shape}")

    print("\nTrain/Test split")
    print("=" * 60)
    print(f"Total ratings: {len(filtered)}")
    print(f"Training ratings: {len(train_data)}")
    print(f"Testing ratings: {len(test_data)}")

    results = evaluate_model(
        test_data,
        train_matrix,
        user_similarity
    )

    print("\nCollaborative Filtering Evaluation")
    print("=" * 60)

    if results is None:
        print("No test ratings could be predicted.")
    else:
        print(f"MAE: {results['MAE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"Coverage: {results['Coverage']:.2%}")
        print(
            f"Predicted test ratings: "
            f"{results['Predictions']} / "
            f"{results['Test-Ratings']}"
        )


if __name__ == "__main__":
    main()