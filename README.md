# Profile-Based Matching System

An intelligent hybrid recommendation system that identifies compatible user profiles using Natural Language Processing (NLP), TF-IDF, cosine similarity, MBTI-based heuristics, demographic attributes, and historical user feedback.

## 1. Project Overview

The Profile-Based Matching System is designed to recommend compatible profiles based on users' professional backgrounds, personal interests, personality preferences, and demographic information.

The system combines multiple matching techniques into a hybrid scoring algorithm and generates the Top 5 recommendations for each selected user.

## 2. Project Objectives

- Generate meaningful user profiles using Python.
- Perform data preprocessing and Exploratory Data Analysis.
- Extract meaningful features from user profile descriptions using NLP.
- Calculate text similarity using TF-IDF and cosine similarity.
- Calculate MBTI and demographic compatibility scores.
- Combine individual scores using a hybrid matching algorithm.
- Use historical feedback to adjust matching weights.
- Generate and rank the Top 5 recommendations.
- Evaluate recommendation performance using Precision@5 and Recall@5.

## 3. Key Features

- Synthetic dataset generation for 100 user profiles.
- NLP-based profile preprocessing.
- TF-IDF text vectorization.
- Cosine similarity calculation.
- MBTI compatibility scoring.
- Location, profession, age, and experience matching.
- Hybrid compatibility scoring.
- Feedback-based weight adjustment.
- Top 5 profile recommendations.
- Streamlit interactive dashboard.
- Accept and Reject feedback functionality.
- Model evaluation using held-out feedback.

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | NLTK, TF-IDF |
| Similarity Measurement | Cosine Similarity |
| Data Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Model Storage | Joblib |
| Testing | Pytest |
| Version Control | Git, GitHub |

## 5. Dataset Description

The project uses synthetically generated datasets.

### users.csv

Contains 100 user profiles with the following attributes:

- user_id
- name
- age
- location
- profession
- experience_years
- professional_summary
- about_me
- mbti
- interests

### feedback.csv

Contains historical user matching decisions.

- user_id
- matched_user_id
- action

Action values:

- 1 – Accepted
- 0 – Rejected

Initial dataset statistics:

| Metric | Value |
|---|---:|
| User profiles | 100 |
| Initial feedback records | 769 |
| Accepted interactions | 538 |
| Rejected interactions | 231 |

The dataset is synthetic and is intended for academic development and evaluation.

## 6. System Architecture

The project follows this processing pipeline:

1. Synthetic Dataset Generation
2. Exploratory Data Analysis
3. NLP Preprocessing
4. TF-IDF Feature Extraction
5. Cosine Similarity Calculation
6. MBTI Compatibility Scoring
7. Demographic Compatibility Scoring
8. Hybrid Matching
9. Feedback-Based Weight Adjustment
10. Top 5 Recommendation Generation
11. Model Evaluation
12. Streamlit Application

## 7. Hybrid Matching Algorithm

The system combines three matching components:

- Text similarity
- MBTI compatibility
- Demographic compatibility

The initial weights are:

| Component | Initial Weight |
|---|---:|
| Text similarity | 50% |
| MBTI compatibility | 20% |
| Demographic compatibility | 30% |

The hybrid compatibility formula is:

**Hybrid Score = (Text Score × 0.50) + (MBTI Score × 0.20) + (Demographic Score × 0.30)**

All component scores are calculated on a 0–100 scale.

### Feedback-Based Weight Adjustment

The system uses historical acceptance and rejection statistics to adjust the initial matching weights.

The learned weights are normalized so that their total equals 100%.

This is an initial heuristic feedback-learning approach. It does not independently identify which matching component caused an individual acceptance or rejection.

## 8. Recommendation Engine

The recommendation engine:

1. Accepts a selected user ID.
2. Calculates compatibility with other profiles.
3. Combines the matching component scores.
4. Applies feedback-adjusted weights.
5. Sorts profiles by compatibility score.
6. Returns the Top 5 recommendations.

The selected user is excluded from their own recommendations.

## 9. Model Evaluation

The system uses an 80:20 interaction-level split of historical feedback.

| Metric | Result |
|---|---:|
| Training records | 615 |
| Testing records | 154 |
| Evaluated users | 63 |
| Precision@5 | 1.59% |
| Recall@5 | 5.03% |
| Test acceptance rate | 70.13% |

These are preliminary results obtained using synthetic data.

The evaluation uses a random interaction-level split. It does not represent a fully independent real-world evaluation, and the results should be interpreted with this limitation.

## 10. Streamlit Application

The interactive application provides:

- User profile selection.
- Selected profile information.
- Top 5 recommended profiles.
- Overall compatibility percentages.
- Individual compatibility score breakdowns.
- Accept and Reject decision selection.
- Feedback submission and CSV storage.

## 11. Project Structure

```text
Profile-Based Matching Algorithm/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── users.csv
│   │   └── feedback.csv
│   └── processed/
│
├── src/
│   └── profile_matching/
│       ├── data/
│       ├── preprocessing/
│       ├── matching/
│       ├── feedback/
│       └── evaluation/
│
├── scripts/
│   └── generate_dataset.py
│
├── tests/
│   └── test_text_processor.py
│
├── reports/
│   └── figures/
│
├── docs/
│
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## 12. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/PranathiR73/profile-based-matching-system.git
cd profile-based-matching-system
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Generate the dataset

```bash
python scripts/generate_dataset.py
```

### Step 5: Run preprocessing

```bash
python -m src.profile_matching.preprocessing.preprocessing_pipeline
```

### Step 6: Run model evaluation

```bash
python -m src.profile_matching.evaluation.model_evaluation
```

### Step 7: Launch the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

### Step 8: Run automated tests

```bash
pytest -v
```

## 13. Testing

The project includes automated tests for:

- Text cleaning
- Tokenization
- Stopword removal
- Lemmatization
- Text preprocessing

Latest verified test result:

**5 tests passed.**

## 14. Limitations

- The system uses synthetic user profiles and feedback.
- MBTI compatibility is a project-specific heuristic, not a scientifically validated compatibility measure.
- Demographic matching uses manually defined scoring rules.
- Feedback-based weight adjustment is an initial heuristic.
- The preliminary evaluation results indicate that recommendation quality needs further improvement.
- Real-world performance requires evaluation using genuine, appropriately collected user feedback.

## 15. Future Enhancements

- Implement advanced NLP using Sentence Transformers.
- Improve feedback learning using individual matching outcomes.
- Experiment with optimized hybrid weights.
- Introduce more comprehensive recommendation evaluation.
- Add user authentication and profile management.
- Improve the user interface and deployment.
- Evaluate the system using real-world feedback.

## 16. Author

Developed as an academic project.

**Project:** Profile-Based Matching System  
**Repository:** https://github.com/PranathiR73/profile-based-matching-system

---

*This project demonstrates the integration of NLP, machine learning techniques, hybrid recommendation methods, and an interactive web application.*