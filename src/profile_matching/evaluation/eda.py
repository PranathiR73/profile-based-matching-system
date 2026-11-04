
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

USERS_PATH = PROJECT_ROOT / "data" / "raw" / "users.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "raw" / "feedback.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Load the user and feedback datasets."""
    users = pd.read_csv(USERS_PATH)
    feedback = pd.read_csv(FEEDBACK_PATH)

    return users, feedback


def analyze_users(users):
    """Analyze user profile information."""

    print("\n========== USER DATASET ANALYSIS ==========")

    print("\nDataset shape:")
    print(users.shape)

    print("\nData types:")
    print(users.dtypes)

    print("\nMissing values:")
    print(users.isnull().sum())

    print("\nDuplicate user IDs:")
    print(users["user_id"].duplicated().sum())

    print("\nAge statistics:")
    print(users["age"].describe())

    print("\nExperience statistics:")
    print(users["experience_years"].describe())

    print("\nProfession distribution:")
    print(users["profession"].value_counts())

    print("\nLocation distribution:")
    print(users["location"].value_counts())

    print("\nMBTI distribution:")
    print(users["mbti"].value_counts())


def analyze_feedback(feedback):
    """Analyze user matching feedback."""

    print("\n========== FEEDBACK DATASET ANALYSIS ==========")

    print("\nDataset shape:")
    print(feedback.shape)

    print("\nMissing values:")
    print(feedback.isnull().sum())

    print("\nDuplicate feedback pairs:")
    print(
        feedback.duplicated(
            subset=["user_id", "matched_user_id"]
        ).sum()
    )

    print("\nFeedback distribution:")
    print(feedback["action"].value_counts().sort_index())

    accepted = (feedback["action"] == 1).sum()
    rejected = (feedback["action"] == 0).sum()
    total = len(feedback)

    print(f"\nTotal interactions: {total}")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Acceptance rate: {accepted / total * 100:.2f}%")


def save_plot(filename):
    """Save the current visualization."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_visualizations(users, feedback):
    create_additional_visualizations(users, feedback)
    """Generate EDA visualizations."""

    sns.set_theme(style="whitegrid")

    # 1. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(users["age"], bins=10, kde=True)
    plt.title("Age Distribution of Users")
    plt.xlabel("Age")
    plt.ylabel("Number of Users")
    save_plot("age_distribution.png")

    # 2. Profession distribution
    plt.figure(figsize=(10, 6))
    users["profession"].value_counts().plot(kind="bar")
    plt.title("Profession Distribution")
    plt.xlabel("Profession")
    plt.ylabel("Number of Users")
    plt.xticks(rotation=45, ha="right")
    save_plot("profession_distribution.png")

    # 3. MBTI distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=users,
        x="mbti",
        order=users["mbti"].value_counts().index
    )
    plt.title("MBTI Personality Distribution")
    plt.xlabel("MBTI Type")
    plt.ylabel("Number of Users")
    plt.xticks(rotation=45)
    save_plot("mbti_distribution.png")

    # 4. Experience distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(users["experience_years"], bins=10, kde=True)
    plt.title("Professional Experience Distribution")
    plt.xlabel("Years of Experience")
    plt.ylabel("Number of Users")
    save_plot("experience_distribution.png")

    # 5. Feedback distribution
    feedback_counts = (
        feedback["action"]
        .map({1: "Accepted", 0: "Rejected"})
        .value_counts()
    )

    plt.figure(figsize=(7, 7))
    plt.pie(
        feedback_counts,
        labels=feedback_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Matching Feedback Distribution")
    save_plot("feedback_distribution.png")

    print("\nAll EDA visualizations saved successfully.")


def main():
    print("PROFILE-BASED MATCHING SYSTEM")
    print("EXPLORATORY DATA ANALYSIS")

    users, feedback = load_datasets()

    analyze_users(users)
    analyze_feedback(feedback)

    create_visualizations(users, feedback)

    print(f"\nVisualizations saved in: {OUTPUT_DIR}")
    print("\nEDA COMPLETED SUCCESSFULLY.")

    
def create_additional_visualizations(users, feedback):
    """Analyze location, profile text and feedback activity."""

    # 6. Location distribution
    plt.figure(figsize=(10, 6))
    users["location"].value_counts().plot(kind="bar")
    plt.title("User Location Distribution")
    plt.xlabel("Location")
    plt.ylabel("Number of Users")
    plt.xticks(rotation=45, ha="right")
    save_plot("location_distribution.png")

    # Calculate word counts in profile descriptions
    users = users.copy()

    users["professional_summary_words"] = (
        users["professional_summary"].astype(str).str.split().str.len()
    )

    users["about_me_words"] = (
        users["about_me"].astype(str).str.split().str.len()
    )

    print("\n========== PROFILE TEXT ANALYSIS ==========")

    print("\nProfessional summary word count:")
    print(users["professional_summary_words"].describe())

    print("\nAbout Me word count:")
    print(users["about_me_words"].describe())

    # 7. Professional summary length
    plt.figure(figsize=(8, 5))
    sns.histplot(users["professional_summary_words"], bins=10, kde=True)
    plt.title("Professional Summary Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Users")
    save_plot("professional_summary_length.png")

    # 8. About Me length
    plt.figure(figsize=(8, 5))
    sns.histplot(users["about_me_words"], bins=10, kde=True)
    plt.title("About Me Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Users")
    save_plot("about_me_length.png")

    # 9. Feedback interactions per user
    interactions = feedback["user_id"].value_counts()

    print("\n========== USER INTERACTION ANALYSIS ==========")
    print(interactions.describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(interactions, bins=8, kde=True)
    plt.title("Feedback Interactions Per User")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Number of Users")
    save_plot("feedback_interactions_per_user.png")

    print("\nAdditional EDA visualizations saved successfully.")


if __name__ == "__main__":
    main()
