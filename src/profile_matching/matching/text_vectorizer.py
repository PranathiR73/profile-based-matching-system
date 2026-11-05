
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.profile_matching.preprocessing.text_processor import TextProcessor


class ProfileTextVectorizer:
    """Convert user profile text into numerical TF-IDF vectors."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.text_processor = TextProcessor()
        self.tfidf_matrix = None
        self.user_ids = None

    def prepare_profile_text(self, df):
        """Combine processed profile descriptions and interests."""

        required_columns = [
            "user_id",
            "processed_professional_summary",
            "processed_about_me",
            "interests"
        ]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        professional = (
            df["processed_professional_summary"].fillna("").astype(str)
        )

        about_me = (
            df["processed_about_me"].fillna("").astype(str)
        )

        interests = df["interests"].fillna("").apply(
            self.text_processor.preprocess
        )

        combined_text = (
            professional + " " + about_me + " " + interests
        ).str.strip()

        if combined_text.eq("").any():
            raise ValueError("Some profiles have empty combined text.")

        return combined_text

    def fit_transform(self, df):
        """Learn TF-IDF vocabulary and transform all user profiles."""

        combined_text = self.prepare_profile_text(df)

        self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)
        self.user_ids = df["user_id"].tolist()

        return self.tfidf_matrix

    def save(self):
        """Save the trained vectorizer and TF-IDF matrix."""

        joblib.dump(
            self.vectorizer,
            "data/processed/tfidf_vectorizer.joblib"
        )

        joblib.dump(
            self.tfidf_matrix,
            "data/processed/profile_tfidf_matrix.joblib"
        )

        joblib.dump(
            self.user_ids,
            "data/processed/profile_user_ids.joblib"
        )


if __name__ == "__main__":

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("TF-IDF TEXT VECTORIZATION")
    print("=" * 60)

    print("\nStep 1: Loading processed dataset...")

    df = pd.read_csv("data/processed/processed_users.csv")

    print("Total profiles loaded:", len(df))

    print("\nStep 2: Preparing profile text...")

    text_vectorizer = ProfileTextVectorizer()
    combined_text = text_vectorizer.prepare_profile_text(df)

    print("Profile text prepared successfully.")

    print("\nStep 3: Applying TF-IDF vectorization...")

    tfidf_matrix = text_vectorizer.fit_transform(df)

    print("\nStep 4: Checking vectorization results...")

    print("Number of profiles:", tfidf_matrix.shape[0])
    print("Number of TF-IDF features:", tfidf_matrix.shape[1])
    print("Matrix shape:", tfidf_matrix.shape)

    print("\nStep 5: Saving trained vectorizer and matrix...")

    text_vectorizer.save()

    print("\nTF-IDF VECTORIZATION COMPLETED SUCCESSFULLY!")
