## Importing and Loading
import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sentence_transformers import SentenceTransformer, util
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from collections import defaultdict

stop_words = set(ENGLISH_STOP_WORDS)

# Loading SBERT Model --- Bigger Model around 400 MB
model = SentenceTransformer('all-mpnet-base-v2')

# Load Resume Data
with open("extracted_resume.json", "r", encoding="utf-8") as f:
    resume_data = json.load(f)

# Load Job Description Data
with open("cleaned_job_description.json", "r", encoding="utf-8") as f:
    jobdesc_data = json.load(f)

# Access Resume Fields
resume_sections = resume_data.get("resume_sections", {})
cleaned_resume_text = resume_data.get("full_cleaned_text", "")

# Access Job Description Fields
semantic_job_desc = jobdesc_data.get("semantic_job_desc", "")
lemmatized_job_desc = jobdesc_data.get("lemmatized_job_desc", [])

# Pre-compute job description embedding (used in section-wise scoring)
job_embedding = model.encode(semantic_job_desc, convert_to_tensor=True)

## Visual Functions
def plot_similarity(score, title):
    """Plot a simple pie chart showing match vs not-match."""
    matched = score
    unmatched = 1 - score
    labels = ['Matched', 'Not Matched']
    sizes = [matched, unmatched]
    colors = ['skyblue', 'lightcoral']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.show()


## Calculating Functions 

#### Compute embedding by using Sbert Model
def compute_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).cpu().numpy()[0][0]


def generate_missing_keywords_wordcloud(cv_text, job_text):
    #Get missing keywords
    missing_keywords = get_missing_keywords(cv_text, job_text)
    
    if not missing_keywords:
        print("✅ No missing keywords found!")
        return

    #Create frequency dictionary (equal weight for all)
    freq_dict = {word: 1 for word in missing_keywords}

    #Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
    #Display the plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Keywords to Add to your CV", fontsize=14)
    plt.tight_layout()
    plt.show()

### Lime Functionality - Lime Explainer

# Custom "classifier" like wrapper for LIME
class SimilarityModel:
    def __init__(self, fixed_text):
        self.fixed_embedding = model.encode(fixed_text, convert_to_tensor=True)

    def predict_proba(self, texts):
        scores = []
        for t in texts:
            emb = model.encode(t, convert_to_tensor=True)
            sim = util.cos_sim(emb, self.fixed_embedding).cpu().numpy()[0][0]
            # Fake binary classes: [similarity, 1-similarity]
            scores.append([sim, 1 - sim])
        return np.array(scores)

def explain_resume_with_lime(resume_text, job_desc_text, num_features=20, num_samples=300):
    ###Use LIME to explain which words affect matching the most and suggest improvements
    sim_model = SimilarityModel(job_desc_text)
    explainer = LimeTextExplainer(class_names=["Match", "No Match"])

    exp = explainer.explain_instance(
        resume_text,
        sim_model.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )
     # Extract features and weights
    features, weights = zip(*exp.as_list())
    
    # Prepare the data for the word cloud (map words to their weights)
    word_weights = {feature: weight for feature, weight in zip(features, weights)}

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_weights)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("LIME Word Contributions (BERT Similarity)")
    plt.tight_layout()
    plt.show()
    
    # Diplaying Lime as Text instead of HTML
    print("\nLIME Important Features (Words Impacting Match):")
    for feature, weight in exp.as_list():
        print(f"{feature:25} --> {weight:+.3f}")

    # --- Suggest Improvements ---
    print("\nSuggestions to Improve Resume:")
    suggestions = []
    for word, weight in exp.as_list():
        if weight < 0:  # Negative influence
            suggestions.append(word)
    
    if suggestions:
        print(f"Try emphasizing or adding keywords like: {', '.join(suggestions)}")
    else:
        print("Resume already matches well! Minor tweaks only.")




########### Lime Functionality for TF-IDF Similarity
# LIME-compatible wrapper for TF-IDF + Cosine Similarity
class TfidfSimilarityModel:
    def __init__(self, fixed_text, vectorizer):
        self.fixed_text = fixed_text
        self.vectorizer = vectorizer
        self.fixed_vector = vectorizer.transform([fixed_text])

    def predict_proba(self, texts):
        scores = []
        for t in texts:
            tfidf_vec = self.vectorizer.transform([t])
            sim = cosine_similarity(tfidf_vec, self.fixed_vector)[0][0]
            scores.append([sim, 1 - sim])  # Fake binary probabilities
        return np.array(scores)

def explain_tfidf_with_lime(resume_text, job_desc_text, vectorizer, num_features=20, num_samples=300):
    """Use LIME to explain TF-IDF-based cosine similarity."""
    sim_model = TfidfSimilarityModel(job_desc_text, vectorizer)
    explainer = LimeTextExplainer(class_names=["Match", "No Match"])

    exp = explainer.explain_instance(
        resume_text,
        sim_model.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )

    # --- Display key features ---
    print("\nLIME Important Features (TF-IDF):")
    for feature, weight in exp.as_list():
        print(f"{feature:25} --> {weight:+.3f}")

    # --- Suggestions to improve resume ---
    print("\nSuggestions to Improve Resume (TF-IDF):")
    suggestions = [word for word, weight in exp.as_list() if weight < 0]
    if suggestions:
        print(f"Consider including or emphasizing: {', '.join(suggestions)}")
    else:
        print("Resume already aligns well! Minor enhancements only.")

# Semantic Similarity with Sbert 
similarity_score = compute_similarity(cleaned_resume_text, semantic_job_desc)
title = "(Resume and Job Description) - SBERT with Cosine Similarity"
# Step 3: Plot Match vs Not Match
plot_similarity(similarity_score,title)



# TF-IDF (Term factor and Inverse  - with cosine similarity)
# Prepare corpus with resume and job description
corpus = [cleaned_resume_text, semantic_job_desc]
# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
# Fit and transform both texts
tfidf_matrix = vectorizer.fit_transform(corpus)
# Compute cosine similarity
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Print result
title = "(Resume vs Job Description) TF-IDF with Cosine Similarity"
plot_similarity(similarity_score, title)

#Compare each resume section and show their matching Score Using - Sbert Cosine
section_scores = {}
for section, text in resume_sections.items():
    if isinstance(text, str) and text.strip():  # Ensure it's a non-empty string
        section_embedding = model.encode(text, convert_to_tensor=True)
        score = util.cos_sim(section_embedding, job_embedding).item()
        section_scores[section] = round(score, 3)

sections = list(section_scores.keys())
scores = [round(score * 100) for score in section_scores.values()]  # Convert to percentage

##### Creating Bar Chart to Show the division of section
plt.figure(figsize=(10, 6))
bars = plt.bar(sections, scores, color='skyblue', edgecolor='black')

# Add percentage labels on top of bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{score}%', ha='center', va='bottom', fontsize=10)
# Styling
plt.ylim(0, 100)
plt.ylabel("Match Score (%)")
plt.title("SBERT - Section-wise Keyword Matching Scores")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Ensure job description is in set format for fast lookup
job_desc_terms = set(lemmatized_job_desc)  # already tokenized and lemmatized

# Prepare results container
matched_keywords_by_section = defaultdict(list)

# Go through each section in the resume
for section_name, section_text in resume_sections.items():
    if not section_text:
        continue

    # Tokenize and clean the section (simple lemmatized-style tokenization)
    section_tokens = re.findall(r'\b\w+\b', section_text.lower())
    
    # Match tokens to job description
    matches = set(section_tokens).intersection(job_desc_terms)
    matched_keywords_by_section[section_name] = sorted(matches)

# --- Display results ---
#print("\n🔍 Overlapping Keywords Between Resume Sections and Job Description:\n")
#for section, matches in matched_keywords_by_section.items():
#    print(f"📄 {section.title()} Section:")
#    print(f"   ➤ {len(matches)} matches: {', '.join(matches)}\n")


# Prepare data
sections = list(matched_keywords_by_section.keys())
match_counts = [len(matched_keywords_by_section[sec]) for sec in sections]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(sections, match_counts, color='mediumseagreen', edgecolor='black')

# Add labels on top of bars
for bar, count in zip(bars, match_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=10)
# Styling the bar chart
plt.ylabel("Number of Keyword Matches")
plt.title("Keyword Overlap - Sectionwise Matching")
plt.ylim(0, max(match_counts) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#### Section Wise - Most Repitiave Word Cloud

# Filter sections with matches
sections_with_matches = {section: keywords for section, keywords in matched_keywords_by_section.items() if keywords}

# Set layout parameters
n_sections = len(sections_with_matches)
cols = min(n_sections, 3)  # Max 3 columns for layout
rows = -(-n_sections // cols)  # Ceiling division

# Create subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = axes.flatten() if n_sections > 1 else [axes]  # Ensure axes is iterable

for ax, (section, keywords) in zip(axes, sections_with_matches.items()):
    freq = {word: keywords.count(word) for word in set(keywords)}
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(freq)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f"{section.title()} Section", fontsize=12, pad=10)
    ax.axis('off')

    # Add border
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             linewidth=2, edgecolor='gray', facecolor='none')
    ax.add_patch(rect)

# Hide any unused axes
for ax in axes[len(sections_with_matches):]:
    ax.axis('off')

plt.tight_layout()
plt.show()

###Keyword matching, Basic keyword Matching

def clean_text(text):
    # Basic text cleaning: lowercasing, remove non-alphanumeric except spaces
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def extract_keywords(text):
    # Very basic keyword extractor: removes stopwords, returns unique words
    words = text.split()
    keywords = set(word for word in words if word not in stop_words and len(word) > 2)
    return keywords

def compute_skills_match(cv_text, job_text):
    # Extract keywords
    cv_keywords = extract_keywords(clean_text(cv_text))
    job_keywords = extract_keywords(clean_text(job_text))

    matched_keywords = cv_keywords.intersection(job_keywords)
    skills_match_score = len(matched_keywords) / max(len(job_keywords), 1)  # Avoid divide by zero

    return skills_match_score, matched_keywords

def get_missing_keywords(cv_text, job_text):
    cv_keywords = extract_keywords(clean_text(cv_text))
    job_keywords = extract_keywords(clean_text(job_text))
    missing_keywords = job_keywords - cv_keywords
    return sorted(missing_keywords)

# Step 2: Skills Match
skills_match_score, matched_skills = compute_skills_match(cleaned_resume_text, semantic_job_desc)

# Combine scores (weighted sum)
#final_score = (0.7 * semantic_similarity) + (0.3 * skills_match_score)

# --- Output ---
#print(f"Semantic Similarity Score: {semantic_similarity:.4f}")
#print(f"Skills Match Score: {skills_match_score:.4f}")
#print(f"Matched Skills: {matched_skills}")
#print(f"\nFinal Combined Match Score: {final_score:.4f}")

generate_missing_keywords_wordcloud(cleaned_resume_text, semantic_job_desc)


# Track all matched keywords across sections
all_matched_keywords = set()
for matches in matched_keywords_by_section.values():
    all_matched_keywords.update(matches)

# Unique keywords in job description
total_job_keywords = len(set(lemmatized_job_desc))

# Compute percentage of job keywords covered
coverage_percent = (len(all_matched_keywords) / total_job_keywords) * 100 if total_job_keywords > 0 else 0

#### Creating Progress Bar
plt.figure(figsize=(8, 1.5))
plt.barh(['Coverage'], [coverage_percent], color='mediumseagreen')
plt.xlim(0, 100)
plt.xlabel("Coverage (%)")
plt.title("Job Description Keyword Coverage")
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add percentage label
plt.text(coverage_percent + 1, 0, f"{coverage_percent:.1f}%", va='center')

plt.tight_layout()
plt.show()

# Explain with LIME and Suggest Improvements (SBERT)
explain_resume_with_lime(cleaned_resume_text, semantic_job_desc, num_features=10, num_samples=200)

# Explain with LIME (TF-IDF)
explain_tfidf_with_lime(cleaned_resume_text, semantic_job_desc, vectorizer, num_features=10, num_samples=200)
