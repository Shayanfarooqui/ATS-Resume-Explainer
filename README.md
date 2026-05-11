# ATS Resume Explainer

An intelligent resume-to-job description matching system that uses Natural Language Processing (NLP) and Explainable AI (XAI) to score how well a resume fits a job posting — and explain *why*.

Built as a Final Year Masters project in Data Science and Analytics.

---

## What It Does

Most Applicant Tracking Systems (ATS) rank resumes automatically without explaining their decisions. This project addresses that by:

1. **Parsing** a resume into structured sections (Skills, Experience, Education, etc.)
2. **Scoring** the match against a job description using two methods:
   - **SBERT** (Sentence-BERT) — semantic similarity using deep learning embeddings
   - **TF-IDF** — keyword frequency-based cosine similarity
3. **Explaining** the score using **LIME** (Local Interpretable Model-agnostic Explanations) — showing which words in the resume positively or negatively affect the match
4. **Visualising** results with pie charts, bar charts, word clouds, and a keyword coverage progress bar

---

## Pipeline Overview

```
1_user_input.py
    └── Loads resume + job description
    └── Saves → user_inputs.json

2_data_extractor.py
    └── Extracts personal info (name, email, phone, links)
    └── Splits resume into sections (Skills, Experience, Education, etc.)
    └── Cleans and lemmatizes job description
    └── Saves → extracted_resume.json
    └── Saves → cleaned_job_description.json

3_semantic_match.py
    └── SBERT cosine similarity (overall match %)
    └── TF-IDF cosine similarity (keyword match %)
    └── Section-wise SBERT scores (bar chart)
    └── Keyword overlap per section (bar chart + word clouds)
    └── Job description keyword coverage (progress bar)
    └── LIME explanation for SBERT (word cloud + feature weights)
    └── LIME explanation for TF-IDF (feature weights)
    └── Missing keywords word cloud (what to add to the CV)
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| `spaCy` (`en_core_web_md`) | NLP — named entity recognition, lemmatization, POS tagging |
| `sentence-transformers` | SBERT semantic embeddings (`all-mpnet-base-v2`) |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `LIME` | Explainability — word-level feature importance |
| `wordcloud` | Visualising keywords |
| `matplotlib` | Pie charts, bar charts, progress bars |
| `geotext` | Location extraction from resume header |

---

## Project Structure

```
ATS-Resume-Explainer/
│
├── 1_user_input.py              # Step 1 – Define resume and job description
├── 2_data_extractor.py          # Step 2 – Parse resume, clean job description
├── 3_semantic_match.py          # Step 3 – Score, visualise, explain
│
├── User Input.ipynb             # Notebook version of Step 1
├── Data Extractor Resume.ipynb  # Notebook version of Step 2
├── Semantic + Skill Keyword Match.ipynb  # Notebook version of Step 3
│
├── user_inputs.json             # Intermediate – raw resume + job description
├── extracted_resume.json        # Intermediate – parsed resume sections
├── cleaned_job_description.json # Intermediate – cleaned + lemmatized job desc
│
├── clean_resume.txt             # Sample resume input
├── clean_jobDesc.txt            # Sample job description input
├── Resume 2.txt                 # Alternative test resume (low-match scenario)
│
├── Final_Project_Report.txt     # Full pipeline as a single reference script
├── .gitignore
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+ and Anaconda (recommended)

```bash
# Clone the repository
git clone https://github.com/Shayanfarooqui/ATS-Resume-Explainer.git
cd ATS-Resume-Explainer

# Install dependencies
pip install spacy geotext lime-stability wordcloud sentence-transformers scikit-learn matplotlib

# Download the spaCy English model
python -m spacy download en_core_web_md
```

> The SBERT model (`all-mpnet-base-v2`, ~420 MB) is downloaded automatically on the first run of Step 3.

---

## Usage

### Option A — Python Scripts (recommended)

Run each step in order from the project directory:

```bash
# Step 1: Set your resume and job description
python 1_user_input.py

# Step 2: Extract resume sections and clean job description
python 2_data_extractor.py

# Step 3: Run matching, visualisation, and LIME explanations
python 3_semantic_match.py
```

### Option B — Jupyter Notebooks

Open each notebook in order and run all cells:

1. `User Input.ipynb`
2. `Data Extractor Resume.ipynb`
3. `Semantic + Skill Keyword Match.ipynb`

### Changing the Resume or Job Description

Edit the `UserInputResume` and `UserInputJobDesc` variables in `1_user_input.py`, then re-run all three steps.

---

## Output Examples

### Overall Match Score (SBERT)
A pie chart showing the percentage of semantic similarity between the resume and job description.

### Section-wise Match Scores
A bar chart scoring each resume section (Skills, Experience, Education, etc.) individually against the job description.

### Keyword Coverage
A progress bar showing what percentage of the job description's key terms are present in the resume.

### LIME Explanation
A ranked list and word cloud of terms in the resume that most positively or negatively influence the match score — with suggestions on keywords to add.

### Missing Keywords Word Cloud
A word cloud of important job description terms not found in the resume.

---

## How LIME Works Here

LIME perturbs the resume text by randomly masking words, then observes how the similarity score changes. Words whose removal causes a large drop in the score are flagged as high-impact positives; words whose removal increases the score are flagged as negatives. This produces a human-readable explanation for an otherwise black-box embedding model.

---

## Limitations

- The SBERT model is large (~420 MB) and slow on CPU — first run takes 1–2 minutes
- LIME uses random sampling, so explanation results vary slightly between runs
- Section extraction relies on standard resume heading names; unconventional formats may not parse correctly
- The LinkedIn and TotalJobs scrapers (in the repo history) were blocked by rate limits and are not part of the current pipeline
