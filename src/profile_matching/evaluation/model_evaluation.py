
import pandas as pd
from sklearn.model_selection import train_test_split

from src.profile_matching.matching.recommendation_engine import (
    RecommendationEngine
)
from src.profile_matching.feedback.feedback_learner import (
    FeedbackLearner
)


class ModelEvaluator:
    """Evaluate recommendations using held-out feedback."""

    def __init__(
        self,
        users_path="data/processed/processed_users.csv",
        feedback_path="data/raw/feedback.csv"
    ):
        self.users_df = pd.read_csv(users_path)
        self.feedback_df = pd.read_csv(feedback_path)

        # Split feedback into training and testing sets.
        self.train_feedback, self.test_feedback = train_test_split(
            self.feedback_df,
            test_size=0.20,
            random_state=42,
            stratify=self.feedback_df["action"]
        )

        # Learn weights using training feedback only.
        self.feedback_learner = FeedbackLearner()
        self.feedback_learner.feedback_df = self.train_feedback.copy()
        self.learned_weights = self.feedback_learner.learn_weights()

        # Initialize recommendation engine.
        self.engine = RecommendationEngine.__new__(
            RecommendationEngine
        )

        self.engine.users_df = self.users_df.copy()
        self.engine.learned_weights = self.learned_weights

        from src.profile_matching.matching.hybrid_matcher import (
            HybridMatcher
        )

        self.engine.matcher = HybridMatcher(self.users_df)
        self.engine.matcher.WEIGHTS = self.learned_weights.copy()

    def evaluate_user(self, user_id, k=5):
        """Evaluate top-K recommendations against test feedback."""

        user_feedback = self.test_feedback[
            self.test_feedback["user_id"] == user_id
        ]

        if user_feedback.empty:
            return None

        accepted_ids = set(
            user_feedback.loc[
                user_feedback["action"] == 1,
                "matched_user_id"
            ]
        )

        # Only evaluate users with at least one positive test example.
        if not accepted_ids:
            return None

        recommendations = self.engine.matcher
        results = []

        for candidate_id in self.users_df["user_id"]:
            if candidate_id == user_id:
                continue

            score = recommendations.calculate_hybrid_score(
                user_id,
                candidate_id
            )

            results.append({
                "user_id": candidate_id,
                "score": score["hybrid_score"]
            })

        ranked = (
            pd.DataFrame(results)
            .sort_values("score", ascending=False)
            .head(k)
        )

        recommended_ids = set(ranked["user_id"])
        relevant = recommended_ids & accepted_ids

        precision = len(relevant) / k
        recall = len(relevant) / len(accepted_ids)

        return {
            "user_id": user_id,
            "precision_at_k": precision,
            "recall_at_k": recall
        }

    def generate_report(self, k=5):
        """Generate aggregate held-out evaluation metrics."""

        results = []

        for user_id in self.test_feedback["user_id"].unique():
            result = self.evaluate_user(user_id, k)

            if result is not None:
                results.append(result)

        results_df = pd.DataFrame(results)

        if results_df.empty:
            raise ValueError(
                "No users with positive test feedback were available."
            )

        accepted = int(
            (self.test_feedback["action"] == 1).sum()
        )

        acceptance_rate = (
            accepted / len(self.test_feedback) * 100
        )

        return {
            "training_records": len(self.train_feedback),
            "testing_records": len(self.test_feedback),
            "evaluated_users": len(results_df),
            "precision_at_k": round(
                results_df["precision_at_k"].mean(), 4
            ),
            "recall_at_k": round(
                results_df["recall_at_k"].mean(), 4
            ),
            "test_acceptance_rate": round(acceptance_rate, 2)
        }


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("HELD-OUT MODEL EVALUATION")
    print("=" * 60)

    evaluator = ModelEvaluator()

    print("\nTraining records:", len(evaluator.train_feedback))
    print("Testing records:", len(evaluator.test_feedback))

    print("\nLearned weights:")
    for component, weight in evaluator.learned_weights.items():
        print(f"{component}: {weight:.4f}")

    print("\nEvaluating held-out recommendations...")

    report = evaluator.generate_report(k=5)

    print("\nHELD-OUT EVALUATION RESULTS")
    print("-" * 40)

    for metric, value in report.items():
        print(f"{metric}: {value}")

    print("\nHELD-OUT MODEL EVALUATION COMPLETED!")
