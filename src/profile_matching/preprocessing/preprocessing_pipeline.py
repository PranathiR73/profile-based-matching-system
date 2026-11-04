
from pathlib import Path
import pandas as pd

from src.profile_matching.preprocessing.text_processor import TextProcessor


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "users.csv"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "processed_users.csv"
)


# ==========================================================
# 2. PREPROCESSING PIPELINE
# ==========================================================

def run_preprocessing_pipeline():
    """Load, preprocess and save all user profiles."""

    print("=" * 60)
    print("PROFILE-BASED MATCHING ALGORITHM")
    print("NLP PREPROCESSING PIPELINE")
    print("=" * 60)

    # Create processor.
    processor = TextProcessor()

    # Load raw dataset.
    print("\nStep 1: Loading raw user dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    users = pd.read_csv(RAW_DATA_PATH)

    print(f"Total profiles loaded: {len(users)}")

    # Check required columns.
    required_columns = [
        "user_id",
        "professional_summary",
        "about_me"
    ]

    for column in required_columns:
        if column not in users.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Preprocess professional summaries.
    print("\nStep 2: Preprocessing professional summaries...")

    users["processed_professional_summary"] = (
        users["professional_summary"].apply(
            processor.preprocess
        )
    )

    # Preprocess personal descriptions.
    print("\nStep 3: Preprocessing personal descriptions...")

    users["processed_about_me"] = (
        users["about_me"].apply(
            processor.preprocess
        )
    )

    # Validate processed data.
    print("\nStep 4: Validating processed dataset...")

    processed_columns = [
        "processed_professional_summary",
        "processed_about_me"
    ]

    if users[processed_columns].isnull().any().any():
        raise ValueError(
            "Processed dataset contains missing values."
        )

    for column in processed_columns:

        if users[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Empty processed text found in {column}."
            )

    # Create output directory.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset.
    print("\nStep 5: Saving processed dataset...")

    users.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # Display results.
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTotal profiles processed: {len(users)}")
    print(f"Total columns: {len(users.columns)}")
    print(f"Output file: {PROCESSED_DATA_PATH}")

    print("\nProcessed dataset preview:")

    print(
        users[
            [
                "user_id",
                "processed_professional_summary",
                "processed_about_me"
            ]
        ].head().to_string(index=False)
    )

    print("\nAll user profiles have been preprocessed successfully!")


# ==========================================================
# 3. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_preprocessing_pipeline()
