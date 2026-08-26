import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system():
    personal_relevance = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "personal_relevance"
    )

    popularity = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "popularity"
    )

    rating_confidence = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "rating_confidence"
    )

    recommendation_score = ctrl.Consequent(
        np.arange(0, 101, 1),
        "recommendation_score"
    )

    # Personal relevance membership functions
    personal_relevance["low"] = fuzz.trimf(
        personal_relevance.universe,
        [0.0, 0.0, 0.5]
    )

    personal_relevance["medium"] = fuzz.trimf(
        personal_relevance.universe,
        [0.2, 0.5, 0.8]
    )

    personal_relevance["high"] = fuzz.trimf(
        personal_relevance.universe,
        [0.5, 1.0, 1.0]
    )

    # Popularity membership functions
    popularity["low"] = fuzz.trimf(
        popularity.universe,
        [0.0, 0.0, 0.5]
    )

    popularity["medium"] = fuzz.trimf(
        popularity.universe,
        [0.2, 0.5, 0.8]
    )

    popularity["high"] = fuzz.trimf(
        popularity.universe,
        [0.5, 1.0, 1.0]
    )

    # Rating confidence membership functions
    rating_confidence["low"] = fuzz.trimf(
        rating_confidence.universe,
        [0.0, 0.0, 0.5]
    )

    rating_confidence["medium"] = fuzz.trimf(
        rating_confidence.universe,
        [0.2, 0.5, 0.8]
    )

    rating_confidence["high"] = fuzz.trimf(
        rating_confidence.universe,
        [0.5, 1.0, 1.0]
    )

    # Recommendation score membership functions
    recommendation_score["low"] = fuzz.trimf(
        recommendation_score.universe,
        [0, 0, 50]
    )

    recommendation_score["medium"] = fuzz.trimf(
        recommendation_score.universe,
        [25, 50, 75]
    )

    recommendation_score["high"] = fuzz.trimf(
        recommendation_score.universe,
        [50, 100, 100]
    )

    rule1 = ctrl.Rule(
        personal_relevance["high"] & popularity["high"],
        recommendation_score["high"]
    )

    rule2 = ctrl.Rule(
        personal_relevance["high"] & rating_confidence["high"],
        recommendation_score["high"]
    )

    rule3 = ctrl.Rule(
        personal_relevance["medium"] & popularity["high"],
        recommendation_score["medium"]
    )

    rule4 = ctrl.Rule(
        personal_relevance["medium"] & rating_confidence["medium"],
        recommendation_score["medium"]
    )

    rule5 = ctrl.Rule(
        personal_relevance["low"],
        recommendation_score["low"]
    )

    rule6 = ctrl.Rule(
        popularity["low"] & rating_confidence["low"],
        recommendation_score["low"]
    )

    recommendation_control = ctrl.ControlSystem(
        [
            rule1,
            rule2,
            rule3,
            rule4,
            rule5,
            rule6
        ]
    )

    recommendation_simulation = ctrl.ControlSystemSimulation(
        recommendation_control
    )

    print("Fuzzy rule system created successfully.")
    return (
        personal_relevance,
        popularity,
        rating_confidence,
        recommendation_score,
        recommendation_simulation
    )

def evaluate_fuzzy_recommendation(
    simulation,
    personal_relevance,
    popularity,
    rating_confidence
):
    simulation.input["personal_relevance"] = personal_relevance
    simulation.input["popularity"] = popularity
    simulation.input["rating_confidence"] = rating_confidence

    simulation.compute()

    return simulation.output["recommendation_score"]

def main():
    (
        personal_relevance,
        popularity,
        rating_confidence,
        recommendation_score,
        simulation
    ) = create_fuzzy_system()

    test_cases = [
        ("High", 0.8, 0.7, 0.8),
        ("Medium", 0.5, 0.6, 0.5),
        ("Low", 0.2, 0.2, 0.2),
    ]

    for name, relevance, popularity_value, confidence in test_cases:
        score = evaluate_fuzzy_recommendation(
            simulation,
            personal_relevance=relevance,
            popularity=popularity_value,
            rating_confidence=confidence
        )

        print(
            f"{name} Case -> "
            f"Relevance: {relevance}, "
            f"Popularity: {popularity_value}, "
            f"Confidence: {confidence}, "
            f"Score: {score:.2f}"
        )


if __name__ == "__main__":
    main()