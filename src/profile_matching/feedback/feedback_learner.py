
import pandas as pd

from pathlib import Path


class FeedbackLearner:
    """Adjust hybrid matching weights using historical user feedback."""

    DEFAULT_WEIGHTS = {
        "text": 0.50,
        "mbti": 0.20,
        "demographic": 0.30
    }

    def __init__(self, feedback_path="data/raw/feedback.csv"):
        self.feedback_path = Path(feedback_path)
        self.feedback_df = None
        self.weights = self.DEFAULT_WEIGHTS.copy()

    def load_feedback(self):
        """Load and validate historical feedback."""

        if not self.feedback_path.exists():
            raise FileNotFoundError(
                f"Feedback file not found: {self.feedback_path}"
            )

        self.feedback_df = pd.read_csv(self.feedback_path)

        required_columns = {
            "user_id",
            "matched_user_id",
            "action"
        }

        missing_columns = required_columns - set(self.feedback_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing feedback columns: {sorted(missing_columns)}"
            )

        if self.feedback_df.empty:
            raise ValueError("Feedback dataset is empty.")

        if not self.feedback_df["action"].isin([0, 1]).all():
            raise ValueError("Feedback action must contain only 0 or 1.")

        if (
            self.feedback_df["user_id"]
            == self.feedback_df["matched_user_id"]
        ).any():
            raise ValueError("Self-match feedback is not allowed.")

        return self.feedback_df

    def calculate_feedback_statistics(self):
        """Calculate acceptance and rejection statistics."""

        if self.feedback_df is None:
            self.load_feedback()

        accepted = int((self.feedback_df["action"] == 1).sum())
        rejected = int((self.feedback_df["action"] == 0).sum())
        total = accepted + rejected

        return {
            "total_feedback": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / total * 100, 2),
            "rejection_rate": round(rejected / total * 100, 2)
        }

    def learn_weights(self):
        """
        Adjust weights using overall acceptance and rejection balance.

        This is an initial heuristic. It does not identify which individual
        matching component caused an acceptance or rejection.
        """

        statistics = self.calculate_feedback_statistics()

        acceptance_rate = statistics["acceptance_rate"] / 100

        # Keep changes small and predictable.
        adjustment = (acceptance_rate - 0.50) * 0.10

        new_weights = {
            "text": self.DEFAULT_WEIGHTS["text"] + adjustment,
            "mbti": self.DEFAULT_WEIGHTS["mbti"] + adjustment * 0.20,
            "demographic": (
                self.DEFAULT_WEIGHTS["demographic"] - adjustment * 1.20
            )
        }

        # Prevent negative weights.
        new_weights = {
            name: max(0.0, weight)
            for name, weight in new_weights.items()
        }

        # Normalize weights so their sum is exactly 1.
        total_weight = sum(new_weights.values())

        self.weights = {
            name: weight / total_weight
            for name, weight in new_weights.items()
        }

        return self.weights

    def get_weights(self):
        """Return the current learned weights."""
        return self.weights.copy()


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("FEEDBACK-BASED LEARNING")
    print("=" * 60)

    learner = FeedbackLearner()

    print("\nStep 1: Loading feedback dataset...")

    feedback = learner.load_feedback()

    print("Feedback records loaded:", len(feedback))

    print("\nStep 2: Calculating feedback statistics...")

    statistics = learner.calculate_feedback_statistics()

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nStep 3: Learning updated matching weights...")

    weights = learner.learn_weights()

    for component, weight in weights.items():
        print(f"{component}: {weight:.4f} ({weight * 100:.2f}%)")

    print("\nTotal weight:", round(sum(weights.values()), 4))

    print("\nFEEDBACK-BASED LEARNING COMPLETED SUCCESSFULLY!")
