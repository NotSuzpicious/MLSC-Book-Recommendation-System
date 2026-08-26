import random
import pandas as pd

from app.optimization.prepare_procurement_data import (
    load_book_features,
    select_procurement_candidates,
    add_procurement_scores,
    add_simulated_costs,
)


BUDGET = 10000
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.02


def prepare_candidates():
    book_features = load_book_features()

    candidates = select_procurement_candidates(
        book_features,
        top_n=100
    )

    candidates = add_procurement_scores(candidates)
    candidates = add_simulated_costs(candidates)

    return candidates.reset_index(drop=True)


def create_individual(candidates):
    individual = [0] * len(candidates)

    indices = list(range(len(candidates)))
    random.shuffle(indices)

    total_cost = 0

    for index in indices:
        cost = candidates.iloc[index]["Cost"]

        if total_cost + cost <= BUDGET:
            # Do not automatically select every affordable book
            if random.random() < 0.5:
                individual[index] = 1
                total_cost += cost

    return individual


def calculate_fitness(individual, candidates):
    total_cost = 0
    total_score = 0

    for index, selected in enumerate(individual):
        if selected == 1:
            total_cost += candidates.iloc[index]["Cost"]
            total_score += candidates.iloc[index]["Procurement-Score"]

    if total_cost > BUDGET:
        return 0

    return total_score

def select_parent(population, candidates):
    tournament = random.sample(population, 3)

    tournament.sort(
        key=lambda individual: calculate_fitness(
            individual,
            candidates
        ),
        reverse=True
    )

    return tournament[0]


def crossover(parent1, parent2):
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    point = random.randint(
        1,
        len(parent1) - 1
    )

    child1 = (
        parent1[:point]
        + parent2[point:]
    )

    child2 = (
        parent2[:point]
        + parent1[point:]
    )

    return child1, child2


def mutate(individual):
    mutated = individual.copy()

    for index in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[index] = 1 - mutated[index]

    return mutated

def repair_individual(individual, candidates):
    repaired = individual.copy()

    total_cost = sum(
        candidates.iloc[index]["Cost"]
        for index, selected in enumerate(repaired)
        if selected == 1
    )

    if total_cost <= BUDGET:
        return repaired

    selected_indices = [
        index
        for index, selected in enumerate(repaired)
        if selected == 1
    ]

    random.shuffle(selected_indices)

    for index in selected_indices:
        if total_cost <= BUDGET:
            break

        repaired[index] = 0
        total_cost -= candidates.iloc[index]["Cost"]

    return repaired

def run_genetic_algorithm(candidates):
    population = [
        create_individual(candidates)
        for _ in range(POPULATION_SIZE)
    ]

    best_individual = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(
                population,
                candidates
            )

            parent2 = select_parent(
                population,
                candidates
            )

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutate(child1)
            child2 = mutate(child2)

            child1 = repair_individual(
                child1,
                candidates
            )

            child2 = repair_individual(
                child2,
                candidates
            )

            new_population.extend(
                [child1, child2]
            )

        population = new_population[:POPULATION_SIZE]

        generation_best = max(
            population,
            key=lambda individual: calculate_fitness(
                individual,
                candidates
            )
        )

        generation_fitness = calculate_fitness(
            generation_best,
            candidates
        )

        if generation_fitness > best_fitness:
            best_fitness = generation_fitness
            best_individual = generation_best.copy()

        if (generation + 1) % 10 == 0:
            print(
                f"Generation {generation + 1}: "
                f"Best Fitness = {best_fitness:.4f}"
            )

    return best_individual, best_fitness

def summarize_solution(best_individual, candidates):
    selected_books = candidates[
        [
            selected == 1
            for selected in best_individual
        ]
    ].copy()

    total_cost = selected_books["Cost"].sum()
    total_score = selected_books["Procurement-Score"].sum()

    selected_books = selected_books.sort_values(
        by="Procurement-Score",
        ascending=False
    )

    return selected_books, total_cost, total_score

def main():
    candidates = prepare_candidates()

    best_individual, best_fitness = run_genetic_algorithm(
        candidates
    )

    selected_books, total_cost, total_score = summarize_solution(
        best_individual,
        candidates
    )

    print("\nGenetic Algorithm completed.")
    print("=" * 70)

    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Selected Books: {len(selected_books)}")
    print(f"Total Cost: ₹{total_cost}")
    print(f"Remaining Budget: ₹{BUDGET - total_cost}")

    print("\nRecommended Books for Procurement:")
    print("=" * 70)

    print(
        selected_books[
            [
                "Book-Title",
                "Book-Author",
                "Procurement-Score",
                "Cost"
            ]
        ]
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()