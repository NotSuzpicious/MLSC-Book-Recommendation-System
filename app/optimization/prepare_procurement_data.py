import pandas as pd
import numpy as np

FEATURES_PATH = "data/processed/book_features.csv"

def add_procurement_scores(candidates):
    candidates = candidates.copy()

    # Log normalization handles highly skewed count data
    candidates["Demand-Score"] = (
        np.log1p(candidates["Interaction-Count"])
        / np.log1p(candidates["Interaction-Count"].max())
    )

    candidates["Confidence-Score"] = (
        np.log1p(candidates["Explicit-Rating-Count"])
        / np.log1p(candidates["Explicit-Rating-Count"].max())
    )

    candidates["Rating-Score"] = (
        candidates["Average-Rating"] / 10.0
    )

    # Overall procurement utility
    candidates["Procurement-Score"] = (
        0.5 * candidates["Demand-Score"]
        + 0.2 * candidates["Confidence-Score"]
        + 0.3 * candidates["Rating-Score"]
    )

    return candidates

def add_simulated_costs(candidates):
    candidates = candidates.copy()

    # Deterministic simulated cost based on ISBN
    # Keeps results reproducible for the project.
    candidates["Cost"] = candidates["ISBN"].apply(
        lambda isbn: 200 + (sum(ord(ch) for ch in str(isbn)) % 601)
    )

    return candidates

def load_book_features():
    return pd.read_csv(
        FEATURES_PATH,
        low_memory=False
    )


def select_procurement_candidates(book_features, top_n=100):
    candidates = book_features[
        book_features["Explicit-Rating-Count"] >= 20
    ].copy()

    candidates = candidates.sort_values(
        by=[
            "Interaction-Count",
            "Average-Rating"
        ],
        ascending=[False, False]
    )

    return candidates.head(top_n)


def main():
    book_features = load_book_features()

    candidates = select_procurement_candidates(
        book_features,
        top_n=100
    )

    candidates = add_procurement_scores(candidates)
    candidates = add_simulated_costs(candidates)

    print("Procurement candidates prepared.")
    print("=" * 70)

    print(f"Candidate books: {len(candidates)}")

    print(
        candidates[
            [
                "ISBN",
                "Book-Title",
                "Interaction-Count",
                "Average-Rating",
                "Procurement-Score",
                "Cost"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()