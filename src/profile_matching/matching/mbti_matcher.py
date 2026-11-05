
class MBTICompatibility:
    """Calculate heuristic compatibility between MBTI types."""

    VALID_TYPES = {
        "ISTJ", "ISFJ", "INFJ", "INTJ",
        "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP",
        "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    }

    def calculate_compatibility(self, mbti_a, mbti_b):
        """Return a heuristic compatibility score from 0 to 100."""

        mbti_a = str(mbti_a).strip().upper()
        mbti_b = str(mbti_b).strip().upper()

        if mbti_a not in self.VALID_TYPES:
            raise ValueError(f"Invalid MBTI type: {mbti_a}")

        if mbti_b not in self.VALID_TYPES:
            raise ValueError(f"Invalid MBTI type: {mbti_b}")

        # Compare the four MBTI dimensions.
        matching_dimensions = sum(
            a == b for a, b in zip(mbti_a, mbti_b)
        )

        # Project-specific heuristic:
        # Each matching dimension contributes 25 points.
        score = matching_dimensions * 25

        return score

    def get_matching_dimensions(self, mbti_a, mbti_b):
        """Return the number of matching MBTI dimensions."""

        mbti_a = str(mbti_a).strip().upper()
        mbti_b = str(mbti_b).strip().upper()

        if mbti_a not in self.VALID_TYPES:
            raise ValueError(f"Invalid MBTI type: {mbti_a}")

        if mbti_b not in self.VALID_TYPES:
            raise ValueError(f"Invalid MBTI type: {mbti_b}")

        return sum(a == b for a, b in zip(mbti_a, mbti_b))


if __name__ == "__main__":

    matcher = MBTICompatibility()

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("MBTI COMPATIBILITY SCORING")
    print("=" * 60)

    test_cases = [
        ("INTJ", "INTJ"),
        ("INTJ", "INTP"),
        ("INTJ", "ENFP"),
        ("ISTJ", "ESFP"),
        ("ENFP", "ENTP")
    ]

    print("\nTesting MBTI compatibility:\n")

    for mbti_a, mbti_b in test_cases:

        score = matcher.calculate_compatibility(
            mbti_a, mbti_b
        )

        matching = matcher.get_matching_dimensions(
            mbti_a, mbti_b
        )

        print(
            f"{mbti_a} vs {mbti_b}: "
            f"{score}% compatibility "
            f"({matching}/4 matching dimensions)"
        )

    print("\nMBTI COMPATIBILITY TESTING COMPLETED!")
