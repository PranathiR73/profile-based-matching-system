
import pandas as pd

from src.profile_matching.matching.hybrid_matcher import HybridMatcher
from src.profile_matching.feedback.feedback_learner import FeedbackLearner


class RecommendationEngine:
    """Generate and rank the top matching user profiles."""

    def __init__(self, users_df):
        self.users_df = users_df.copy()

        # Learn weights from historical feedback.
        self.feedback_learner = FeedbackLearner()
        self.learned_weights = self.feedback_learner.learn_weights()

        # Initialize the hybrid matcher.
        self.matcher = HybridMatcher(self.users_df)

        # Apply the learned weights.
        self.matcher.WEIGHTS = self.learned_weights.copy()

    def recommend(self, user_id, top_n=5):
        """Return the highest-ranked profiles for a selected user."""

        if user_id not in self.users_df["user_id"].values:
            raise ValueError(f"User not found: {user_id}")

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        recommendations = []

        for matched_user_id in self.users_df["user_id"]:
            if matched_user_id == user_id:
                continue

            result = self.matcher.calculate_hybrid_score(
                user_id,
                matched_user_id
            )

            profile = self.users_df[
                self.users_df["user_id"] == matched_user_id
            ].iloc[0]

            recommendations.append({
                "user_id": matched_user_id,
                "name": profile["name"],
                "profession": profile["profession"],
                "location": profile["location"],
                "text_score": result["text_score"],
                "mbti_score": result["mbti_score"],
                "demographic_score": result["demographic_score"],
                "compatibility_score": result["hybrid_score"]
            })

        recommendations_df = pd.DataFrame(recommendations)

        recommendations_df = recommendations_df.sort_values(
            by="compatibility_score",
            ascending=False
        ).reset_index(drop=True)

        return recommendations_df.head(top_n)


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("TOP 5 RECOMMENDATION ENGINE")
    print("=" * 60)

    print("\nStep 1: Loading user profiles...")

    users_df = pd.read_csv("data/processed/processed_users.csv")

    print("Total profiles:", len(users_df))

    print("\nStep 2: Initializing recommendation engine...")

    engine = RecommendationEngine(users_df)

    print("Learned weights:", engine.learned_weights)

    print("\nStep 3: Generating top 5 recommendations for U001...")

    recommendations = engine.recommend("U001", top_n=5)

    print("\nTOP 5 RECOMMENDED PROFILES")
    print("-" * 80)

    print(
        recommendations[
            [
                "user_id",
                "name",
                "profession",
                "location",
                "compatibility_score"
            ]
        ].to_string(index=False)
    )

    print("\nRECOMMENDATION ENGINE COMPLETED SUCCESSFULLY!")
