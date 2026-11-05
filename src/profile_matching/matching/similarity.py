
import joblib

from sklearn.metrics.pairwise import cosine_similarity


class ProfileSimilarity:
    """Calculate similarity between user profile vectors."""

    def __init__(self):
        self.tfidf_matrix = None
        self.user_ids = None
        self.similarity_matrix = None

    def load_data(self):
        """Load saved TF-IDF matrix and user IDs."""

        self.tfidf_matrix = joblib.load(
            "data/processed/profile_tfidf_matrix.joblib"
        )

        self.user_ids = joblib.load(
            "data/processed/profile_user_ids.joblib"
        )

    def calculate_similarity(self):
        """Calculate pairwise cosine similarity."""

        if self.tfidf_matrix is None:
            raise ValueError("Load TF-IDF data before calculating similarity.")

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        return self.similarity_matrix

    def get_similar_users(self, user_id, top_n=5):
        """Return the most similar users, excluding the selected user."""

        if self.similarity_matrix is None:
            raise ValueError("Calculate similarity before requesting matches.")

        if user_id not in self.user_ids:
            raise ValueError(f"User ID not found: {user_id}")

        user_index = self.user_ids.index(user_id)

        scores = self.similarity_matrix[user_index]

        ranked_indices = scores.argsort()[::-1]

        recommendations = []

        for index in ranked_indices:
            if index == user_index:
                continue

            recommendations.append({
                "user_id": self.user_ids[index],
                "similarity_score": round(float(scores[index]), 4)
            })

            if len(recommendations) == top_n:
                break

        return recommendations


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("COSINE SIMILARITY")
    print("=" * 60)

    similarity = ProfileSimilarity()

    print("\nStep 1: Loading TF-IDF data...")
    similarity.load_data()
    print("TF-IDF data loaded successfully.")

    print("\nStep 2: Calculating cosine similarity...")
    matrix = similarity.calculate_similarity()

    print("Similarity matrix shape:", matrix.shape)

    print("\nStep 3: Checking similarity values...")
    print("Minimum similarity:", round(float(matrix.min()), 4))
    print("Maximum similarity:", round(float(matrix.max()), 4))

    print("\nStep 4: Finding top 5 similar users for U001...")

    recommendations = similarity.get_similar_users("U001", top_n=5)

    for rank, recommendation in enumerate(recommendations, start=1):
        print(
            f"{rank}. {recommendation['user_id']} - "
            f"Similarity: {recommendation['similarity_score']}"
        )

    print("\nCOSINE SIMILARITY COMPLETED SUCCESSFULLY!")
