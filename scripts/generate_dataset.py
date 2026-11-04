
"""
PROJECT: PROFILE-BASED MATCHING ALGORITHM

MODULE: Synthetic Dataset Generation

Generates:
1. users.csv
2. feedback.csv

Dataset characteristics:
- 100 synthetic user profiles
- 5 to 10 feedback interactions per user
- Valid MBTI personality types
- Professional and personal descriptions
- Accept (1) / Reject (0) feedback

All generated records are synthetic and intended
for experimentation and academic evaluation.
"""

from pathlib import Path
import random
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_SEED = 42

TOTAL_USERS = 100

MIN_INTERACTIONS = 5
MAX_INTERACTIONS = 10

random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. PROFILE DATA
# ==================================================

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


# ==================================================
# 3. PROFESSIONAL SUMMARY TEMPLATES
# ==================================================

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
        "Has {experience_duration} of professional experience and "
        "enjoys continuous learning. Interested in {goal}."
    ),

    (
        "A technology professional specializing in {field}. "
        "Has {experience_duration} of experience as a {profession}. "
        "Aims to contribute to innovative projects and {goal}."
    )
]


# ==================================================
# 4. PERSONAL DESCRIPTION TEMPLATES
# ==================================================

PERSONAL_TEMPLATES = [

    (
        "I enjoy learning new things and exploring technology. "
        "In my free time, I like {hobby1} and {hobby2}. "
        "I value teamwork, honesty, and meaningful conversations."
    ),

    (
        "I am a curious and motivated person who enjoys "
        "{hobby1}. I also spend time on {hobby2}. "
        "I appreciate creativity, personal growth, and collaboration."
    ),

    (
        "I enjoy meeting people with different perspectives. "
        "My interests include {hobby1} and {hobby2}. "
        "I believe in continuous improvement and maintaining a "
        "positive attitude."
    ),

    (
        "Outside work, I enjoy {hobby1} and {hobby2}. "
        "I like exchanging ideas, learning from others, "
        "and working toward meaningful goals."
    )
]


# ==================================================
# 5. GENERATE USER PROFILES
# ==================================================

def generate_users(total_users: int) -> pd.DataFrame:
    """
    Generate synthetic user profiles.

    Returns:
        DataFrame containing user profile information.
    """

    users = []

    for user_number in range(1, total_users + 1):

        # Generate basic personal information
        first_name = random.choice(FIRST_NAMES)

        last_name = random.choice(LAST_NAMES)

        profession = random.choice(PROFESSIONS)

        experience = random.randint(0, 10)

        # Correct singular and plural experience wording
        experience_duration = (
            f"{experience} year"
            if experience == 1
            else f"{experience} years"
        )

        # Generate professional information
        field = random.choice(INTEREST_OPTIONS)

        career_goal = random.choice(CAREER_GOALS)

        professional_summary = random.choice(
            PROFESSIONAL_TEMPLATES
        ).format(
            profession=profession,
            experience_duration=experience_duration,
            field=field,
            goal=career_goal
        )

        # Generate personal information
        hobby1, hobby2 = random.sample(HOBBIES, 2)

        about_me = random.choice(
            PERSONAL_TEMPLATES
        ).format(
            hobby1=hobby1,
            hobby2=hobby2
        )

        # Generate interests
        interests = random.sample(
            INTEREST_OPTIONS,
            k=random.randint(3, 5)
        )

        # Construct profile
        user = {
            "user_id": f"U{user_number:03d}",
            "name": f"{first_name} {last_name}",
            "age": random.randint(21, 40),
            "location": random.choice(LOCATIONS),
            "profession": profession,
            "experience_years": experience,
            "professional_summary": professional_summary,
            "about_me": about_me,
            "mbti": random.choice(MBTI_TYPES),
            "interests": ", ".join(interests)
        }

        users.append(user)

    return pd.DataFrame(users)


# ==================================================
# 6. GENERATE FEEDBACK DATASET
# ==================================================

def generate_feedback(users: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic accept/reject feedback.

    action:
        1 = Accept
        0 = Reject

    Every user receives 5 to 10 interactions.
    """

    feedback_records = []

    user_ids = users["user_id"].tolist()

    for user_id in user_ids:

        # Exclude the current user from possible matches
        possible_matches = [
            candidate
            for candidate in user_ids
            if candidate != user_id
        ]

        interaction_count = random.randint(
            MIN_INTERACTIONS,
            MAX_INTERACTIONS
        )

        # Select unique matching candidates
        selected_matches = random.sample(
            possible_matches,
            k=interaction_count
        )

        for matched_user_id in selected_matches:

            # Synthetic feedback distribution
            action = random.choices(
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


# ==================================================
# 7. VALIDATE GENERATED DATASETS
# ==================================================

def validate_datasets(
    users: pd.DataFrame,
    feedback: pd.DataFrame
) -> None:
    """
    Check important dataset quality requirements.

    Raises:
        ValueError if validation fails.
    """

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

    # Check required columns
    if not all(
        column in users.columns
        for column in required_user_columns
    ):
        raise ValueError("Missing required user dataset columns.")

    if not all(
        column in feedback.columns
        for column in required_feedback_columns
    ):
        raise ValueError("Missing required feedback dataset columns.")

    # Check empty values
    if users[required_user_columns].isnull().any().any():
        raise ValueError("User dataset contains missing values.")

    if feedback[required_feedback_columns].isnull().any().any():
        raise ValueError("Feedback dataset contains missing values.")

    # Check empty text fields
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

    # Check unique user IDs
    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs found.")

    # Check valid MBTI values
    if not users["mbti"].isin(MBTI_TYPES).all():
        raise ValueError("Invalid MBTI personality type found.")

    # Check valid feedback actions
    if not feedback["action"].isin([0, 1]).all():
        raise ValueError("Feedback action must be 0 or 1.")

    # Check feedback references
    valid_user_ids = set(users["user_id"])

    if not feedback["user_id"].isin(valid_user_ids).all():
        raise ValueError("Unknown user ID found in feedback.")

    if not feedback["matched_user_id"].isin(valid_user_ids).all():
        raise ValueError("Unknown matched user ID found.")

    # Users cannot match themselves
    if (feedback["user_id"] == feedback["matched_user_id"]).any():
        raise ValueError("Self-matching records found.")

    # Check interaction count per user
    interactions = feedback.groupby("user_id").size()

    if not interactions.between(
        MIN_INTERACTIONS,
        MAX_INTERACTIONS
    ).all():
        raise ValueError(
            "Each user must have between 5 and 10 interactions."
        )

    # Check duplicate feedback records
    if feedback.duplicated(
        subset=["user_id", "matched_user_id"]
    ).any():
        raise ValueError("Duplicate feedback interactions found.")

    print("\nAll dataset validation checks passed!")


# ==================================================
# 8. SAVE DATASETS
# ==================================================

def main():
    """Execute the complete dataset generation pipeline."""

    print("=" * 55)
    print("PROFILE-BASED MATCHING ALGORITHM")
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 55)

    # Generate users
    print("\nGenerating user profiles...")

    users = generate_users(TOTAL_USERS)

    # Generate feedback
    print("Generating matching feedback...")

    feedback = generate_feedback(users)

    # Validate before saving
    print("\nValidating datasets...")

    validate_datasets(users, feedback)

    # Save datasets
    users_path = DATA_DIR / "users.csv"

    feedback_path = DATA_DIR / "feedback.csv"

    users.to_csv(users_path, index=False)

    feedback.to_csv(feedback_path, index=False)

    # Display results
    print("\n" + "=" * 55)
    print("DATASET GENERATION COMPLETED")
    print("=" * 55)

    print(f"\nUsers dataset: {users_path}")

    print(f"Feedback dataset: {feedback_path}")

    print("\nDATASET SUMMARY")

    print(f"Total users: {len(users)}")

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
        ].head(5).to_string(index=False)
    )

    print("\nFeedback dataset preview:")

    print(feedback.head(5).to_string(index=False))

    print("\nInteractions per user:")

    print(
        feedback.groupby("user_id")
        .size()
        .describe()
    )

    print("\nSynthetic dataset generation finished successfully.")


# ==================================================
# 9. PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
