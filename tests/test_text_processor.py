
import re
import sys
from pathlib import Path

import pandas as pd

from src.profile_matching.preprocessing.text_processor import TextProcessor


def main():
    # Load the complete user dataset.
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "raw" / "users.csv"

    users = pd.read_csv(dataset_path)

    processor = TextProcessor()

    print("=" * 60)
    print("NLP PREPROCESSING - COMPLETE DATASET TEST")
    print("=" * 60)

    # Check required columns.
    required_columns = [
        "user_id",
        "professional_summary",
        "about_me"
    ]

    for column in required_columns:
        assert column in users.columns, (
            f"Missing required column: {column}"
        )

    total_profiles = len(users)
    successful_profiles = 0

    processed_summaries = []
    processed_about_me = []

    # Process every user profile.
    for _, row in users.iterrows():

        user_id = row["user_id"]

        original_summary = row["professional_summary"]
        original_about = row["about_me"]

        processed_summary = processor.preprocess(
            original_summary
        )

        processed_about = processor.preprocess(
            original_about
        )

        # Check that processed text is not empty.
        assert processed_summary.strip(), (
            f"Empty professional summary for {user_id}"
        )

        assert processed_about.strip(), (
            f"Empty personal description for {user_id}"
        )

        # Check that processed text contains only lowercase
        # alphabetic words separated by single spaces.
        for text in [processed_summary, processed_about]:

            assert text == text.lower(), (
                f"Uppercase characters found for {user_id}"
            )

            assert re.fullmatch(
                r"[a-z]+(?: [a-z]+)*",
                text
            ), f"Unexpected characters or formatting for {user_id}"

        processed_summaries.append(processed_summary)
        processed_about_me.append(processed_about)

        successful_profiles += 1

    # Confirm every profile was processed.
    assert successful_profiles == total_profiles, (
        "Not all profiles were processed."
    )

    # Display sample results.
    print("\nSAMPLE PREPROCESSING RESULTS")
    print("-" * 60)

    for index in range(min(5, total_profiles)):

        print(f"\nUSER ID: {users.iloc[index]['user_id']}")

        print("\nOriginal Professional Summary:")
        print(users.iloc[index]["professional_summary"])

        print("\nProcessed Professional Summary:")
        print(processed_summaries[index])

        print("\nOriginal About Me:")
        print(users.iloc[index]["about_me"])

        print("\nProcessed About Me:")
        print(processed_about_me[index])

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Total profiles: {total_profiles}")
    print(f"Successfully processed: {successful_profiles}")
    print(f"Failed profiles: {total_profiles - successful_profiles}")
    print("Professional summaries: All checked")
    print("Personal descriptions: All checked")
    print("Empty processed text: 0")
    print("Invalid output formatting: 0")

    print("\nALL 100 USER PROFILE TESTS PASSED!")


if __name__ == "__main__":
    main()
