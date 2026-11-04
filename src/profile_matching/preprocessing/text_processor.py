
import re

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextProcessor:
    """Clean and preprocess profile text for NLP matching."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        # Preserve meaningful words used in professional profiles.
        self.stop_words.discard("full")

    def clean_text(self, text):
        """Convert text to lowercase and remove unwanted characters."""

        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Keep alphabetic characters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        """Split cleaned text into individual words."""

        return text.split()

    def remove_stopwords(self, tokens):
        """Remove common English stopwords."""

        return [
            word for word in tokens
            if word not in self.stop_words
        ]

    def lemmatize(self, tokens):
        """Convert words into their base forms."""

        return [
            self.lemmatizer.lemmatize(word)
            for word in tokens
        ]

    def preprocess(self, text):
        """Execute the complete NLP preprocessing pipeline."""

        cleaned_text = self.clean_text(text)

        tokens = self.tokenize(cleaned_text)

        filtered_tokens = self.remove_stopwords(tokens)

        lemmatized_tokens = self.lemmatize(filtered_tokens)

        return " ".join(lemmatized_tokens)


if __name__ == "__main__":

    processor = TextProcessor()

    sample_text = (
        "I am interested in developing innovative AI "
        "applications and building intelligent systems."
    )

    print("ORIGINAL TEXT:")
    print(sample_text)

    print("\nPREPROCESSED TEXT:")
    print(processor.preprocess(sample_text))
