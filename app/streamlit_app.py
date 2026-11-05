
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile_matching.matching.recommendation_engine import (
    RecommendationEngine
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Profile-Based Matching System",
    page_icon="🤝",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

USERS_PATH = PROJECT_ROOT / "data" / "processed" / "processed_users.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "raw" / "feedback.csv"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_users():
    return pd.read_csv(USERS_PATH)


def load_feedback():
    return pd.read_csv(FEEDBACK_PATH)


def create_recommendation_engine(users_df):
    return RecommendationEngine(users_df)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🤝 Profile-Based Matching System")

st.markdown(
    """
    Discover compatible profiles using an intelligent hybrid
    recommendation algorithm powered by NLP, MBTI heuristics,
    demographic compatibility and historical feedback.
    """
)

st.divider()


# ============================================================
# VALIDATE DATA
# ============================================================

if not USERS_PATH.exists():
    st.error("Processed user dataset not found.")
    st.stop()

if not FEEDBACK_PATH.exists():
    st.error("Feedback dataset not found.")
    st.stop()

users_df = load_users()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Profile Selection")

selected_user_id = st.sidebar.selectbox(
    "Select your profile",
    users_df["user_id"].tolist()
)

selected_profile = users_df[
    users_df["user_id"] == selected_user_id
].iloc[0]

st.sidebar.subheader("Your Profile")

st.sidebar.write(f"**Name:** {selected_profile['name']}")
st.sidebar.write(f"**Profession:** {selected_profile['profession']}")
st.sidebar.write(f"**Location:** {selected_profile['location']}")
st.sidebar.write(f"**Age:** {selected_profile['age']}")
st.sidebar.write(
    f"**Experience:** {selected_profile['experience_years']} years"
)
st.sidebar.write(f"**MBTI:** {selected_profile['mbti']}")

st.sidebar.caption(
    "Recommendations use project-specific compatibility heuristics."
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Profiles", len(users_df))

with col2:
    st.metric("Matching Components", 3)

with col3:
    st.metric("Recommendations", 5)

st.divider()


# ============================================================
# DISPLAY SAVED MESSAGE
# ============================================================

if "feedback_saved_message" in st.session_state:

    st.success(st.session_state["feedback_saved_message"])

    del st.session_state["feedback_saved_message"]


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

st.header("Top 5 Recommended Profiles")

if st.button("Generate Recommendations", type="primary"):

    engine = create_recommendation_engine(users_df)

    recommendations = engine.recommend(
        selected_user_id,
        top_n=5
    )

    st.session_state["recommendations"] = recommendations
    st.session_state["selected_user_id"] = selected_user_id

    # Remove old decisions when generating a new recommendation list.
    for key in list(st.session_state.keys()):
        if key.startswith("decision_"):
            del st.session_state[key]


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if (
    "recommendations" in st.session_state
    and st.session_state.get("selected_user_id") == selected_user_id
):

    recommendations = st.session_state["recommendations"]

    if recommendations.empty:

        st.warning("No recommendations available.")

    else:

        for rank, (_, recommendation) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            matched_user_id = recommendation["user_id"]

            with st.container(border=True):

                st.subheader(
                    f"#{rank} {recommendation['name']}"
                )

                st.write(
                    f"**Profession:** {recommendation['profession']}"
                )

                st.write(
                    f"**Location:** {recommendation['location']}"
                )

                score = float(
                    recommendation["compatibility_score"]
                )

                st.metric(
                    "Overall Compatibility",
                    f"{score:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, score / 100))
                )

                with st.expander("View compatibility breakdown"):

                    st.write(
                        f"Text similarity: "
                        f"{recommendation['text_score']:.2f}%"
                    )

                    st.write(
                        f"MBTI compatibility: "
                        f"{recommendation['mbti_score']:.2f}%"
                    )

                    st.write(
                        f"Demographic compatibility: "
                        f"{recommendation['demographic_score']:.2f}%"
                    )

                # One dropdown for each recommended profile.
                feedback_key = (
                    f"decision_{selected_user_id}_{matched_user_id}"
                )

                st.selectbox(
                    "Your decision",
                    options=[
                        "Not selected",
                        "Accept",
                        "Reject"
                    ],
                    key=feedback_key
                )


# ============================================================
# SAVE FEEDBACK
# ============================================================

st.divider()

st.header("Submit Your Feedback")

if st.button("Save Feedback"):

    if (
        "recommendations" not in st.session_state
        or st.session_state.get("selected_user_id") != selected_user_id
    ):

        st.warning(
            "Generate recommendations before submitting feedback."
        )

    else:

        recommendations = st.session_state["recommendations"]

        feedback_records = []

        for _, recommendation in recommendations.iterrows():

            matched_user_id = recommendation["user_id"]

            feedback_key = (
                f"decision_{selected_user_id}_{matched_user_id}"
            )

            decision = st.session_state.get(
                feedback_key,
                "Not selected"
            )

            if decision == "Accept":
                action = 1

            elif decision == "Reject":
                action = 0

            else:
                continue

            feedback_records.append({
                "user_id": selected_user_id,
                "matched_user_id": matched_user_id,
                "action": action
            })

        if not feedback_records:

            st.warning(
                "Select Accept or Reject for at least one profile."
            )

        else:

            try:

                existing_feedback = load_feedback()

                new_feedback = pd.DataFrame(feedback_records)

                # Remove previous records for the same user-profile pairs.
                pairs_to_update = set(
                    zip(
                        new_feedback["user_id"],
                        new_feedback["matched_user_id"]
                    )
                )

                existing_pairs = list(
                    zip(
                        existing_feedback["user_id"],
                        existing_feedback["matched_user_id"]
                    )
                )

                keep_mask = [
                    pair not in pairs_to_update
                    for pair in existing_pairs
                ]

                existing_feedback = existing_feedback.loc[
                    keep_mask
                ]

                # Combine old and new feedback.
                combined_feedback = pd.concat(
                    [existing_feedback, new_feedback],
                    ignore_index=True
                )

                # Save to CSV.
                combined_feedback.to_csv(
                    FEEDBACK_PATH,
                    index=False
                )

                # Clear previous recommendations.
                st.session_state.pop("recommendations", None)
                st.session_state.pop("selected_user_id", None)

                # Clear decision widget states.
                for key in list(st.session_state.keys()):

                    if key.startswith("decision_"):
                        del st.session_state[key]

                # Store success message for the next page run.
                st.session_state["feedback_saved_message"] = (
                    f"Successfully saved {len(feedback_records)} "
                    "feedback records."
                )

                # Refresh the application.
                st.rerun()

            except (OSError, PermissionError, ValueError) as error:

                st.error(f"Unable to save feedback: {error}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Profile-Based Matching System | NLP + Hybrid Recommendation "
    "| Academic Project"
)
