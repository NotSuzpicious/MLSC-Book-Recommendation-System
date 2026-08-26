import streamlit as st
import random

from app.optimization.weight_optimizer import (
    prepare_candidates,
    run_genetic_algorithm,
    summarize_solution,
    RANDOM_SEED,
)
from app.services.recommendation_service import (
    prepare_recommendation_system,
    get_recommendations_for_user,
    get_fallback_recommendations,
)

st.set_page_config(
    page_title="Smart Library Recommendation System",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #d9d9d9;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_recommendation_system():
    return prepare_recommendation_system()

st.title("Smart Library Recommendation and Resource Allocation System")

st.caption(
    "Machine Learning and Soft Computing Mini Project"
)

st.markdown(
    """
    This system provides personalized book recommendations for readers
    and supports library-side procurement decisions using optimization.
    """
)

st.divider()
page = st.sidebar.radio(
    "Select Module",
    [
        "Book Recommendations",
        "Procurement Optimization"
    ]
)

if page == "Book Recommendations":

    st.subheader("Personalized Book Recommendation")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        step=1,
        value=242
    )

    top_n = st.selectbox(
        "Number of recommendations",
        options=[5, 10],
        index=0
    )

    if st.button("Get Recommendations"):

        with st.spinner("Preparing recommendations..."):
            (
                books,
                book_features,
                filtered,
                matrix,
                user_similarity,
                tfidf_matrix
            ) = load_recommendation_system()

            recommendations = get_recommendations_for_user(
                int(user_id),
                books,
                book_features,
                filtered,
                matrix,
                user_similarity,
                tfidf_matrix,
                top_n=int(top_n)
            )

        if recommendations:

            st.subheader("Recommended Books")

            for rank, item in enumerate(
                recommendations,
                start=1
            ):
                st.markdown(
                    f"""
                    **{rank}. {item['Book-Title']}**  
                    Author: {item['Book-Author']}  
                    Recommendation Score: {item['Fuzzy-Score']:.2f}
                    """
                )

        else:

            st.warning(
                "Personalized recommendations are not available "
                "for this user. Showing popular books instead."
            )

            fallback = get_fallback_recommendations(
                book_features,
                top_n=int(top_n)
            )

            for rank, item in enumerate(
                fallback,
                start=1
            ):
                st.markdown(
                    f"""
                    **{rank}. {item['Book-Title']}**  
                    Author: {item['Book-Author']}  
                    Average Rating: {item['Average-Rating']:.2f}
                    """
                )


elif page == "Procurement Optimization":

    st.subheader("Library Procurement Optimization")

    st.write(
        "Use a Genetic Algorithm to identify books "
        "that should be prioritized for procurement "
        "under a limited budget."
    )

    if st.button("Run Procurement Optimization"):

        with st.spinner("Running Genetic Algorithm..."):

            random.seed(RANDOM_SEED)

            candidates = prepare_candidates()

            best_individual, best_fitness = (
                run_genetic_algorithm(candidates)
            )

            (
                selected_books,
                total_cost,
                total_score
            ) = summarize_solution(
                best_individual,
                candidates
            )

        st.success(
            "Optimization completed successfully."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Selected Books",
            len(selected_books)
        )

        col2.metric(
            "Total Cost",
            f"₹{total_cost}"
        )

        col3.metric(
            "Best Fitness",
            f"{best_fitness:.4f}"
        )

        st.subheader(
            "Recommended Books for Procurement"
        )

        st.dataframe(
            selected_books[
                [
                    "Book-Title",
                    "Book-Author",
                    "Procurement-Score",
                    "Cost"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )