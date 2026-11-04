
"""
PROJECT: PROFILE-BASED MATCHING ALGORITHM

MODULE: Synthetic Dataset Generation

Generates:
    1. data/raw/users.csv
    2. data/raw/feedback.csv

Dataset:
    - 100 synthetic user profiles
    - 5 to 10 feedback interactions per user
    - All 16 valid MBTI types
    - Professional and personal descriptions
    - Accept (1) / Reject (0) feedback

All records are synthetic and intended for academic
experimentation and evaluation.
"""

from pathlib import Path
import random
import re

import pandas as pd


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

RANDOM_SEED = 42
TOTAL_USERS = 100

MIN_INTERACTIONS = 5
MAX_INTERACTIONS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. PROFILE DATA
# ==========================================================

FIRST_NAMES = [
    "Aarav", "Aditi", "Ananya", "Arjun", "Diya",
    "Ishaan", "Kavya", "Meera", "Nikhil", "Priya",
    "Rahul", "Riya", "Sanjay", "Sneha", "Varun",
    "Vikram", "Neha", "Kiran", "Rohan", "Shreya"
]

LAST_NAMES = [
    "Sharma", "Reddy", "Patel", "Kumar", "Rao",
    "Shetty", "Verma", "Iyer", "Nair", "Joshi",
    "Desai", "Kulkarni", "Mehta", "Gowda", "Das"
]

LOCATIONS = [
    "Bengaluru", "Mysuru", "Hyderabad", "Chennai",
    "Pune", "Mumbai", "Delhi", "Kochi",
    "Mangaluru", "Coimbatore"
]

PROFESSIONS = [
    "Software Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "AI Engineer",
    "Data Analyst",
    "Product Manager",
    "UI/UX Designer",
    "Business Analyst",
    "Cybersecurity Analyst",
    "Cloud Engineer",
    "Research Analyst",
    "Full Stack Developer"
]

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

INTEREST_OPTIONS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science",
    "Python Programming",
    "Web Development",
    "Cloud Computing",
    "Cybersecurity",
    "Research",
    "Entrepreneurship",
    "Leadership",
    "Reading",
    "Photography",
    "Painting",
    "Music",
    "Travel",
    "Sports",
    "Technology",
    "Problem Solving",
    "Product Development",
    "Design Thinking"
]

CAREER_GOALS = [
    "developing scalable AI applications",
    "becoming an expert in data science",
    "leading innovative technology projects",
    "building intelligent software solutions",
    "advancing research in machine learning",
    "developing secure and reliable systems",
    "creating useful products for real-world problems",
    "improving skills in cloud technologies"
]

HOBBIES = [
    "reading technical articles",
    "painting",
    "playing chess",
    "listening to music",
    "photography",
    "travelling",
    "coding personal projects",
    "playing badminton",
    "cooking",
    "watching documentaries",
    "learning new languages",
    "exploring new technologies"
]


# ==========================================================
# 3. PROFESSIONAL SUMMARY TEMPLATES
# ==========================================================

PROFESSIONAL_TEMPLATES = [
    (
        "A {profession} with {experience_duration} of experience "
        "working on {field}. Interested in building practical "
        "solutions and improving technical expertise. "
        "Currently focused on {goal}."
    ),
    (
        "Working as a {profession} with experience in {field}. "
        "Enjoys solving complex problems and collaborating with "
        "cross-functional teams. Career goal is {goal}."
    ),
    (
        "An aspiring {profession} passionate about {field}. "
        "Has {experience_duration} of professional experience "
        "and enjoys continuous learning. Interested in {goal}."
    ),
    (
        "A technology professional specializing in {field}. "
        "Has {experience_duration} of experience as a {profession}. "
        "Aims to contribute to innovative projects and {goal}."
    )
]


# ==========================================================
# 4. PERSONAL DESCRIPTION TEMPLATES
# ==========================================================

PERSONAL_TEMPLATES = [
    (
        "I enjoy learning new things and exploring technology. "
        "In my free time, I like {hobby1} and {hobby2}. "
        "I value teamwork, honesty, and meaningful conversations."
    ),
    (
        "I am a curious and motivated person who enjoys {hobby1}. "
        "I also spend time on {hobby2}. "
        "I appreciate creativity, personal growth, and collaboration."
    ),
    (
        "I enjoy meeting people with different perspectives. "
        "My interests include {hobby1} and {hobby2}. "
        "I believe in continuous improvement and maintaining "
        "a positive attitude."
    ),
    (
        "Outside work, I enjoy {hobby1} and {hobby2}. "
        "I like exchanging ideas, learning from others, "
        "and working toward meaningful goals."
    )
]


# ==========================================================
# 5. TEXT NORMALIZATION
# ==========================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation spacing."""

    text = str(text)

    # Add missing spaces after sentence punctuation.
    text = re.sub(
        r"([.!?])(?=[A-Za-z])",
        r"\1 ",
        text
    )

    # Add missing spaces after commas.
    text = re.sub(
        r",(?=\S)",
        ", ",
        text
    )

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([.,!?])",
        r"\1",
        text
    )

    return text.strip()


# ==========================================================
# 6. GENERATE USER PROFILES
# ==========================================================

def generate_users(total_users: int, rng) -> pd.DataFrame:
    """Generate synthetic user profiles."""

    users = []

    for user_number in range(1, total_users + 1):

        first_name = rng.choice(FIRST_NAMES)
        last_name = rng.choice(LAST_NAMES)

        profession = rng.choice(PROFESSIONS)
        experience = rng.randint(0, 10)

        if experience == 1:
            experience_duration = "1 year"
        else:
            experience_duration = f"{experience} years"

        field = rng.choice(INTEREST_OPTIONS)
        career_goal = rng.choice(CAREER_GOALS)

        # Generate professional summary.
        professional_template = rng.choice(
            PROFESSIONAL_TEMPLATES
        )

        professional_summary = professional_template.format(
            profession=profession,
            experience_duration=experience_duration,
            field=field,
            goal=career_goal
        )

        # Generate personal description.
        hobby1, hobby2 = rng.sample(HOBBIES, 2)

        personal_template = rng.choice(
            PERSONAL_TEMPLATES
        )

        about_me = personal_template.format(
            hobby1=hobby1,
            hobby2=hobby2
        )

        # Normalize descriptions.
        professional_summary = normalize_text(
            professional_summary
        )

        about_me = normalize_text(about_me)

        # Generate interests.
        interests = rng.sample(
            INTEREST_OPTIONS,
            k=rng.randint(3, 5)
        )

        user = {
            "user_id": f"U{user_number:03d}",
            "name": f"{first_name} {last_name}",
            "age": rng.randint(21, 40),
            "location": rng.choice(LOCATIONS),
            "profession": profession,
            "experience_years": experience,
            "professional_summary": professional_summary,
            "about_me": about_me,
            "mbti": rng.choice(MBTI_TYPES),
            "interests": ", ".join(interests)
        }

        users.append(user)

    return pd.DataFrame(users)


# ==========================================================
# 7. GENERATE FEEDBACK DATASET
# ==========================================================

def generate_feedback(users: pd.DataFrame, rng) -> pd.DataFrame:
    """Generate synthetic accept/reject feedback."""

    feedback_records = []

    user_ids = users["user_id"].tolist()

    for user_id in user_ids:

        possible_matches = [
            candidate
            for candidate in user_ids
            if candidate != user_id
        ]

        interaction_count = rng.randint(
            MIN_INTERACTIONS,
            MAX_INTERACTIONS
        )

        selected_matches = rng.sample(
            possible_matches,
            k=interaction_count
        )

        for matched_user_id in selected_matches:

            action = rng.choices(
                population=[1, 0],
                weights=[0.65, 0.35],
                k=1
            )[0]

            feedback_records.append({
                "user_id": user_id,
                "matched_user_id": matched_user_id,
                "action": action
            })

    return pd.DataFrame(feedback_records)


# ==========================================================
# 8. VALIDATE GENERATED DATASETS
# ==========================================================

def validate_datasets(
    users: pd.DataFrame,
    feedback: pd.DataFrame
) -> None:
    """Validate dataset quality before saving."""

    required_user_columns = [
        "user_id",
        "name",
        "age",
        "location",
        "profession",
        "experience_years",
        "professional_summary",
        "about_me",
        "mbti",
        "interests"
    ]

    required_feedback_columns = [
        "user_id",
        "matched_user_id",
        "action"
    ]

    # Check required columns.
    if not all(
        column in users.columns
        for column in required_user_columns
    ):
        raise ValueError("Missing required user columns.")

    if not all(
        column in feedback.columns
        for column in required_feedback_columns
    ):
        raise ValueError("Missing required feedback columns.")

    # Check number of profiles.
    if len(users) != TOTAL_USERS:
        raise ValueError(
            f"Expected {TOTAL_USERS} users, found {len(users)}."
        )

    # Check missing values.
    if users[required_user_columns].isnull().any().any():
        raise ValueError("User dataset contains missing values.")

    if feedback[required_feedback_columns].isnull().any().any():
        raise ValueError("Feedback dataset contains missing values.")

    # Check empty text values.
    text_columns = [
        "name",
        "location",
        "profession",
        "professional_summary",
        "about_me",
        "interests"
    ]

    for column in text_columns:
        if users[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Empty text values found in {column}."
            )

    # Check unique user IDs.
    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs found.")

    # Check valid MBTI values.
    if not users["mbti"].isin(MBTI_TYPES).all():
        raise ValueError("Invalid MBTI personality type found.")

    # Check age and experience.
    if not users["age"].between(21, 40).all():
        raise ValueError("Invalid age values found.")

    if not users["experience_years"].between(0, 10).all():
        raise ValueError("Invalid experience values found.")

    # Check description punctuation and known joined phrases.
    invalid_phrases = [
        "ofprofessional",
        "yearof",
        "yearsof",
        "leadinginnovative",
        "solvingcomplex",
        "enjoyssolving",
        "meaningfulconversations",
        "buildingintelligent",
        "softwaresolutions",
        "exchangingideas",
        "currentlyfocused",
        "interestedin",
        "careergoal"
    ]

    for column in ["professional_summary", "about_me"]:

        if users[column].str.contains(
            r"\s+[.,!?]",
            regex=True
        ).any():
            raise ValueError(
                f"Incorrect punctuation spacing in {column}."
            )

        if users[column].str.contains(
            r"[.!?][A-Z]",
            regex=True
        ).any():
            raise ValueError(
                f"Missing sentence spacing in {column}."
            )

        for phrase in invalid_phrases:
            if users[column].str.contains(
                re.escape(phrase),
                case=False,
                regex=True
            ).any():
                raise ValueError(
                    f"Possible joined words '{phrase}' found "
                    f"in {column}."
                )

    # Check feedback actions.
    if not feedback["action"].isin([0, 1]).all():
        raise ValueError("Feedback action must be 0 or 1.")

    # Check feedback references.
    valid_user_ids = set(users["user_id"])

    if not feedback["user_id"].isin(valid_user_ids).all():
        raise ValueError("Unknown user ID in feedback.")

    if not feedback["matched_user_id"].isin(valid_user_ids).all():
        raise ValueError("Unknown matched user ID in feedback.")

    # Check self-matching.
    if (
        feedback["user_id"] == feedback["matched_user_id"]
    ).any():
        raise ValueError("Self-matching records found.")

    # Check duplicate feedback pairs.
    if feedback.duplicated(
        subset=["user_id", "matched_user_id"]
    ).any():
        raise ValueError("Duplicate feedback records found.")

    # Check interaction counts.
    interactions = feedback.groupby("user_id").size()

    if len(interactions) != len(users):
        raise ValueError(
            "Every user must have feedback interactions."
        )

    if not interactions.between(
        MIN_INTERACTIONS,
        MAX_INTERACTIONS
    ).all():
        raise ValueError(
            "Each user must have between 5 and 10 interactions."
        )

    print("\nAll dataset validation checks passed!")


# ==========================================================
# 9. SAVE DATASETS
# ==========================================================

def main():
    """Execute the complete dataset generation pipeline."""

    rng = random.Random(RANDOM_SEED)

    print("=" * 55)
    print("PROFILE-BASED MATCHING ALGORITHM")
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 55)

    print("\nGenerating user profiles...")
    users = generate_users(TOTAL_USERS, rng)

    print("Generating matching feedback...")
    feedback = generate_feedback(users, rng)

    print("\nValidating datasets...")
    validate_datasets(users, feedback)

    users_path = DATA_DIR / "users.csv"
    feedback_path = DATA_DIR / "feedback.csv"

    # Save only after successful validation.
    users.to_csv(users_path, index=False)
    feedback.to_csv(feedback_path, index=False)

    print("\n" + "=" * 55)
    print("DATASET GENERATION COMPLETED")
    print("=" * 55)

    print(f"\nUsers dataset: {users_path}")
    print(f"Feedback dataset: {feedback_path}")

    print(f"\nTotal users: {len(users)}")
    print(f"Total feedback records: {len(feedback)}")

    print("\nFeedback distribution:")
    print(
        feedback["action"]
        .value_counts()
        .sort_index()
        .rename(index={
            0: "Rejected",
            1: "Accepted"
        })
    )

    print("\nUsers dataset preview:")
    print(
        users[
            [
                "user_id",
                "name",
                "profession",
                "experience_years",
                "mbti"
            ]
        ].head().to_string(index=False)
    )

    print("\nFirst five professional summaries:")
    for _, row in users.head().iterrows():
        print(row["user_id"], "-", row["professional_summary"])

    print("\nFirst five personal descriptions:")
    for _, row in users.head().iterrows():
        print(row["user_id"], "-", row["about_me"])

    print("\nInteractions per user:")
    print(
        feedback.groupby("user_id")
        .size()
        .describe()
    )

    print("\nSynthetic dataset generation finished successfully.")


# ==========================================================
# 10. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
