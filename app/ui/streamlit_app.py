import random
import html

import streamlit as st

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


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Library",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* ---------------------------------------------------
       GLOBAL
    --------------------------------------------------- */

    .stApp {
        background: #FFFDF7;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* IMPORTANT:
       Explicitly force readable text on light background.
    */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li,
    .stApp div {
        color: #29241E;
    }

    h1, h2, h3 {
        color: #29241E !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-weight: 750;
    }

    h2, h3 {
        font-weight: 650;
    }

    hr {
        border: none;
        border-top: 1px solid #E9E1D3;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }


    /* ---------------------------------------------------
       SIDEBAR
    --------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #172033;
        border-right: 1px solid #38322B;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFDF7 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.55rem 0.75rem;
        border-radius: 9px;
        margin-bottom: 0.25rem;
    }


    /* ---------------------------------------------------
       BUTTONS
    --------------------------------------------------- */

    .stButton > button {
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.25rem;
        background: #F4B942;
        color: #29241E !important;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    .stButton > button p,
    .stButton > button span {
        color: #29241E !important;
    }

    .stButton > button:hover {
        background: #D99A18;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(217, 154, 24, 0.22);
    }

    .stButton > button:hover p,
    .stButton > button:hover span {
        color: #FFFFFF !important;
    }


    /* ---------------------------------------------------
       INPUTS
    --------------------------------------------------- */

    div[data-baseweb="input"] {
        background: white;
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #111827 !important;
    }

    div[data-baseweb="select"] > div {
        background: white;
        border-radius: 10px;
        color: #111827 !important;
    }


    /* ---------------------------------------------------
       METRIC CARDS
    --------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E9E1D3;
        border-top: 3px solid #F4B942;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(41, 36, 30, 0.05);
    }

    div[data-testid="stMetricLabel"] * {
        color: #746B60 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #29241E !important;
    }


    /* ---------------------------------------------------
       REUSABLE CARDS
    --------------------------------------------------- */

    .feature-card {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 16px;
        padding: 24px;
        min-height: 180px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .feature-card h3 {
        margin-top: 0;
        margin-bottom: 8px;
    }

    .feature-icon {
        font-size: 28px;
        margin-bottom: 12px;
    }


    /* ---------------------------------------------------
       RECOMMENDATION CARDS
    --------------------------------------------------- */

    .book-card {
        background: #FFFFFF;
        border: 1px solid #E9E1D3;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(41, 36, 30, 0.04);
    }

    .book-rank {
        display: inline-block;
        background: #FFF4CC;
        color: #9A6700 !important;
        font-weight: 700;
        border-radius: 8px;
        padding: 5px 9px;
        margin-bottom: 10px;
    }

    .book-title {
        color: #111827 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .book-author {
        color: #64748b !important;
        font-size: 14px;
        margin-bottom: 12px;
    }

    .score-badge {
        display: inline-block;
        background: #EDF8F2;
        color: #2E7D5B !important;
        border: 1px solid #B9DFC9;
        border-radius: 8px;
        padding: 5px 9px;
        font-size: 13px;
        font-weight: 650;
    }


    /* ---------------------------------------------------
       TECHNIQUE BADGES
    --------------------------------------------------- */

    .tech-badge {
        display: inline-block;
        padding: 7px 11px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        background: #FFF4CC;
        border: 1px solid #F1D784;
        color: #805B10 !important;
        font-size: 13px;
        font-weight: 600;
    }


    /* ---------------------------------------------------
       CONTROL PANEL
    --------------------------------------------------- */

    .control-title {
        font-size: 14px;
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }


    /* ---------------------------------------------------
       PROCUREMENT TABLE
    --------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }


    /* ---------------------------------------------------
       SMALL TEXT
    --------------------------------------------------- */

    .muted-text {
        color: #64748b !important;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------------

@st.cache_resource
def load_recommendation_system():
    return prepare_recommendation_system()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 Smart Library")
    st.caption("ML & Soft Computing")
    st.caption("Decision Support System")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Project Overview",
            "Book Recommendations",
            "Procurement Optimization"
        ]
    )

    st.divider()

    st.caption(
        "Personalized recommendations and "
        "resource planning for smarter libraries."
    )


# ---------------------------------------------------------
# GLOBAL HEADER
# ---------------------------------------------------------

st.title("Smart Library Recommendation & Resource Allocation")

st.caption(
    "An intelligent decision-support system for readers and library administrators."
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.markdown(
        """
        The system combines **machine learning**, **fuzzy logic**, and
        **optimization** to improve book discovery while also supporting
        smarter library procurement decisions.
        """
    )

    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📖</div>
                <h3>Personalized Recommendations</h3>
                <p>
                    Discover books using collaborative filtering,
                    content-based filtering, hybrid recommendation,
                    and fuzzy ranking.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Smart Resource Allocation</h3>
                <p>
                    Help library administrators prioritize valuable
                    books for procurement using a Genetic Algorithm
                    under a constrained budget.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Techniques Used")

    st.markdown(
        """
        <span class="tech-badge">Collaborative Filtering</span>
        <span class="tech-badge">Content-Based Filtering</span>
        <span class="tech-badge">Hybrid Recommendation</span>
        <span class="tech-badge">Fuzzy Logic</span>
        <span class="tech-badge">Genetic Algorithm</span>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Model Evaluation")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Personalized Hit Rate@10",
        "22.90%"
    )

    metric2.metric(
        "Recommendation Coverage",
        "36.60%"
    )

    metric3.metric(
        "CF Prediction Coverage",
        "19.91%"
    )

    st.markdown("")

    metric4, metric5, metric6 = st.columns(3)

    metric4.metric(
        "Popularity Hit Rate@10",
        "3.10%"
    )

    metric5.metric(
        "Collaborative MAE",
        "1.6151"
    )

    metric6.metric(
        "Collaborative RMSE",
        "2.1558"
    )

    st.info(
        "The personalized recommender achieved a substantially higher "
        "Hit Rate@10 than the global popularity baseline."
    )


# =========================================================
# BOOK RECOMMENDATIONS
# =========================================================

elif page == "Book Recommendations":

    st.subheader("Personalized Book Recommendations")

    st.markdown(
        """
        Enter a reader's User ID to generate books tailored to their
        historical preferences and interaction patterns.
        """
    )

    st.divider()

    input_col1, input_col2 = st.columns([2, 1])

    with input_col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            step=1,
            value=242,
            help="Try User 242 or User 11676 for personalized results."
        )

    with input_col2:
        top_n = st.selectbox(
            "Number of recommendations",
            options=[5, 10],
            index=0
        )

    st.markdown("")

    if st.button(
        "Find Recommended Books",
        type="primary"
    ):

        with st.spinner(
            "Analyzing reader preferences and preparing recommendations..."
        ):
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

        st.divider()

        if recommendations:

            st.subheader(f"Recommended for User {int(user_id)}")

            st.caption(
                "Recommendations are ranked using hybrid recommendation "
                "and fuzzy decision-making."
            )

            for rank, item in enumerate(
                recommendations,
                start=1
            ):
                book_row = books[
                    books["ISBN"] == item["ISBN"]
                ]

                if not book_row.empty:
                    cover_url = book_row.iloc[0]["Image-URL-M"]
                else:
                    cover_url = None

                card_col1, card_col2, card_col3 = st.columns(
                    [1, 5, 1.4],
                    vertical_alignment="center"
                )

                with card_col1:
                    if cover_url:
                        st.image(
                            cover_url,
                            width=95
                        )
                    else:
                        st.markdown("📖")

                with card_col2:
                    st.markdown(
                        f"### {rank}. {item['Book-Title']}"
                    )

                    st.caption(
                        f"by {item['Book-Author']}"
                    )

                    st.write(
                        "Personalized using hybrid recommendation "
                        "and fuzzy ranking."
                    )

                with card_col3:
                    st.metric(
                        "Fuzzy Score",
                        f"{item['Fuzzy-Score']:.2f}"
                    )

                st.divider()

        else:

            st.warning(
                "This user does not have sufficient interaction history "
                "for reliable personalized recommendations."
            )

            st.caption(
                "Showing highly rated popular books instead."
            )

            fallback = get_fallback_recommendations(
                book_features,
                top_n=int(top_n)
            )

            for rank, item in enumerate(
                fallback,
                start=1
            ):
                title = html.escape(
                    str(item["Book-Title"])
                )

                author = html.escape(
                    str(item["Book-Author"])
                )

                rating = item["Average-Rating"]

                st.markdown(
                    f"""
                    <div class="book-card">
                        <div class="book-rank">#{rank}</div>
                        <div class="book-title">{title}</div>
                        <div class="book-author">
                            by {author}
                        </div>
                        <span class="score-badge">
                            Average Rating: {rating:.2f} / 10
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# PROCUREMENT OPTIMIZATION
# =========================================================

elif page == "Procurement Optimization":

    st.subheader("Library Procurement Optimization")

    st.markdown(
        """
        Identify books that should be prioritized for acquisition while
        making the best possible use of a limited library procurement budget.
        """
    )

    st.info(
        "The optimizer uses a Genetic Algorithm to search for a "
        "high-value combination of books under the configured budget."
    )

    st.markdown("")

    if st.button(
        "Run Procurement Optimization",
        type="primary"
    ):

        with st.spinner(
            "Running Genetic Algorithm and evaluating procurement plans..."
        ):

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
            "Procurement plan generated successfully."
        )

        st.divider()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Books Selected",
            len(selected_books)
        )

        metric2.metric(
            "Total Procurement Cost",
            f"₹{total_cost:,}"
        )

        metric3.metric(
            "Optimization Fitness",
            f"{best_fitness:.4f}"
        )

        st.divider()

        st.subheader("Recommended Books for Procurement")

        display_table = selected_books[
            [
                "Book-Title",
                "Book-Author",
                "Procurement-Score",
                "Cost"
            ]
        ].copy()

        display_table = display_table.rename(
            columns={
                "Book-Title": "Book Title",
                "Book-Author": "Author",
                "Procurement-Score": "Procurement Score",
                "Cost": "Cost (₹)"
            }
        )

        display_table["Procurement Score"] = (
            display_table["Procurement Score"]
            .round(3)
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The Genetic Algorithm searches for a combination of "
            "high-value books that maximizes procurement utility while "
            "remaining within the available budget."
        )

        st.caption(
            "Note: Procurement costs are simulated for this academic prototype."
        )