
class DemographicMatcher:
    """Calculate demographic compatibility between two profiles."""

    WEIGHTS = {
        "location": 0.25,
        "profession": 0.35,
        "experience": 0.25,
        "age": 0.15
    }

    def calculate_location_score(self, location_a, location_b):
        """Compare user locations."""

        if not location_a or not location_b:
            return 0.0

        return 100.0 if (
            str(location_a).strip().lower()
            == str(location_b).strip().lower()
        ) else 0.0

    def calculate_profession_score(self, profession_a, profession_b):
        """Compare user professions."""

        if not profession_a or not profession_b:
            return 0.0

        return 100.0 if (
            str(profession_a).strip().lower()
            == str(profession_b).strip().lower()
        ) else 0.0

    def calculate_experience_score(self, experience_a, experience_b):
        """Compare professional experience."""

        difference = abs(float(experience_a) - float(experience_b))

        return max(0.0, 100.0 - (difference * 10.0))

    def calculate_age_score(self, age_a, age_b):
        """Compare user ages."""

        difference = abs(float(age_a) - float(age_b))

        return max(0.0, 100.0 - (difference * 5.0))

    def calculate_compatibility(self, profile_a, profile_b):
        """Calculate weighted demographic compatibility."""

        location_score = self.calculate_location_score(
            profile_a["location"],
            profile_b["location"]
        )

        profession_score = self.calculate_profession_score(
            profile_a["profession"],
            profile_b["profession"]
        )

        experience_score = self.calculate_experience_score(
            profile_a["experience_years"],
            profile_b["experience_years"]
        )

        age_score = self.calculate_age_score(
            profile_a["age"],
            profile_b["age"]
        )

        final_score = (
            location_score * self.WEIGHTS["location"]
            + profession_score * self.WEIGHTS["profession"]
            + experience_score * self.WEIGHTS["experience"]
            + age_score * self.WEIGHTS["age"]
        )

        return {
            "location_score": round(location_score, 2),
            "profession_score": round(profession_score, 2),
            "experience_score": round(experience_score, 2),
            "age_score": round(age_score, 2),
            "demographic_score": round(final_score, 2)
        }


if __name__ == "__main__":

    matcher = DemographicMatcher()

    user_a = {
        "location": "Bengaluru",
        "profession": "Data Scientist",
        "experience_years": 5,
        "age": 28
    }

    user_b = {
        "location": "Bengaluru",
        "profession": "Data Scientist",
        "experience_years": 4,
        "age": 30
    }

    print("=" * 60)
    print("PROFILE-BASED MATCHING SYSTEM")
    print("DEMOGRAPHIC COMPATIBILITY SCORING")
    print("=" * 60)

    results = matcher.calculate_compatibility(user_a, user_b)

    for factor, score in results.items():
        print(f"{factor}: {score}")

    print("\nDEMOGRAPHIC SCORING COMPLETED SUCCESSFULLY!")
