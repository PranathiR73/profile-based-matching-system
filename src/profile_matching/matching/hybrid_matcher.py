
import pandas as pd

from src.profile_matching.matching.similarity import ProfileSimilarity
from src.profile_matching.matching.mbti_matcher import MBTICompatibility
from src.profile_matching.matching.demographic_matcher import DemographicMatcher


class HybridMatcher:
    """Combine text, MBTI and demographic scores."""

    WEIGHTS = {
        "text": 0.50,
        "mbti": 0.20,
        "demographic": 0.30
    }

    def __init__(self, users_df):
        self.users_df = users_df.copy()

        self.similarity_matcher = ProfileSimilarity()
        self.mbti_matcher = MBTICompatibility()
        self.demographic_matcher = DemographicMatcher()

        self.similarity_matcher.load_data()
        self.similarity_matcher.calculate_similarity()

        self.users_df = self.users_df.set_index(
            "user_id", drop=False
        )

    def calculate_hybrid_score(self, user_a_id, user_b_id):
        """Calculate final compatibility between two users."""

        if user_a_id == user_b_id:
            raise ValueError("A user cannot be matched with themselves.")

        if user_a_id not in self.users_df.index:
            raise ValueError(f"User not found: {user_a_id}")

        if user_b_id not in self.users_df.index:
            raise ValueError(f"User not found: {user_b_id}")

        profile_a = self.users_df.loc[user_a_id]
        profile_b = self.users_df.loc[user_b_id]

        index_a = self.similarity_matcher.user_ids.index(user_a_id)
        index_b = self.similarity_matcher.user_ids.index(user_b_id)

        text_score = float(
            self.similarity_matcher.similarity_matrix[index_a, index_b]
        ) * 100

        mbti_score = self.mbti_matcher.calculate_compatibility(
            profile_a["mbti"],
            profile_b["mbti"]
        )

        demographic_results = (
            self.demographic_matcher.calculate_compatibility(
                profile_a,
                profile_b
            )
        )

        demographic_score = demographic_results["demographic_score"]

        hybrid_score = (
            text_score * self.WEIGHTS["text"]
            + mbti_score * self.WEIGHTS["mbti"]
            + demographic_score * self.WEIGHTS["demographic"]
        )

        return {
            "user_a": user_a_id,
            "user_b": user_b_id,
            "text_score": round(text_score, 2),
            "mbti_score": round(mbti_score, 2),
            "demographic_score": round(demographic_score, 2),
            "hybrid_score": round(hybrid_score, 2)
        }


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("HYBRID MATCHING ALGORITHM")
    print("=" * 60)

    print("\nStep 1: Loading user dataset...")

    users_df = pd.read_csv("data/processed/processed_users.csv")

    print("Total profiles:", len(users_df))

    print("\nStep 2: Initializing hybrid matcher...")

    matcher = HybridMatcher(users_df)

    print("Hybrid matcher initialized successfully.")

    print("\nStep 3: Calculating compatibility between U001 and U077...")

    result = matcher.calculate_hybrid_score("U001", "U077")

    print("\nCOMPATIBILITY RESULTS")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nHYBRID MATCHING COMPLETED SUCCESSFULLY!")
