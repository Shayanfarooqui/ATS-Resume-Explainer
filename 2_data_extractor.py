import re # Regix for regular expressions
import spacy # NLP Library for text processing
import json
from geotext import GeoText # Importing geotext library to find geo locations


# Load JSON content into a Python dictionary
with open("user_inputs.json", "r", encoding="utf-8") as f:
    resume = json.load(f)
    
clean_resume = resume["resume"]


# Load the spaCy model
nlp = spacy.load("en_core_web_md")


clean_resume

#Finding Location from Resume

def extract_places_from_header(text, lines=5):
    # Get only the first few lines
    header = '\n'.join(text.strip().split('\n')[:lines])
    doc = nlp(header)
    
    # Extract places (GPE = Geo-Political Entity)
    places = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return places


#Function to Lowercase the whole Text
def lowercase_text(text):
    """Convert the entire text to lowercase."""
    return text.lower()


#Finding the name from the first non-empty line or using NER as fallback.
def extract_name(text):
    """Extract the name from the first non-empty line or using NER as fallback."""
    lines = text.strip().split("\n")
    name_line = next((line.strip() for line in lines if line.strip()), None)

    if not name_line:
        return None  # Empty text case

    doc = nlp(name_line)
    if any(ent.label_ == "PERSON" for ent in doc.ents):
        return name_line  # Looks like a name already
    
    # Fallback: Search the full text
    doc_full = nlp(text)
    for ent in doc_full.ents:
        if ent.label_ == "PERSON":
            return ent.text
    
    return None  # No name found

def extract_emails(text):
    """Extract all email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails

def extract_phone_numbers(text):
    """Extract phone numbers (UK style and general) from text."""
    phone_pattern = r'(\+?\d{1,3}[\s-]?\d{3,4}[\s-]?\d{5,6})'
    phones = re.findall(phone_pattern, text)
    return phones

def extract_links(text):
    """Extract web links including LinkedIn, even if no https."""
    # Pattern to match normal URLs and LinkedIn profiles, case-insensitive
    link_pattern = r'(https?://[^\s|]+|www\.[^\s|]+|(?:linkedin|github|bitbucket)\.com/[^\s|]+)'
    links = re.findall(link_pattern, text, flags=re.IGNORECASE)
    return links


#Removing from Original Text Functions

def clean_resume_text(text, name=None, emails=None, phones=None, links=None, places=None):
    """Remove extracted name, emails, phones, links, and places from the text."""
    cleaned_text = text

    # Remove name
    if name:
        cleaned_text = cleaned_text.replace(name, '')
    
    # Remove emails
    if emails:
        for e in emails:
            cleaned_text = cleaned_text.replace(e, '')
    
    # Remove phones
    if phones:
        for p in phones:
            cleaned_text = cleaned_text.replace(p, '')
    
    # Remove links
    if links:
        for l in links:
            cleaned_text = cleaned_text.replace(l, '')
    
    # Remove places (using regex to handle word boundaries and case-insensitivity)
    if places:
        for place in places:
            pattern = r'\b' + re.escape(place) + r'\b'
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    return cleaned_text


#text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text) # To remove dates from the string
def clean_special_characters(text):
    # Replace all non-alphanumeric characters (except newline) with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Execution Area
name = extract_name(clean_resume)
emails = extract_emails(clean_resume)
phones = extract_phone_numbers(clean_resume)
places = extract_places_from_header(clean_resume, lines=5)
links = extract_links(clean_resume)

print("Name:", name)
print("Email(s):", emails)
print("Phone Number(s):", phones)
print("Places:", places)
print("Link(s):", links)


cleaned_text = clean_resume_text(
    clean_resume,
    name=name,
    emails=emails,
    phones=phones,
    links=links,
    places=places
)


cleaned_text

# Defining Resume section headers
RESUME_SECTIONS = [
    "Contact Information", "Objective", "Summary", "Education", "Experience", "Skills", "Projects",
    "Certifications", "Licenses", "Awards", "Honors", "Publications", "References", "Technical Skills","External Courses",
    "Computer Skills", "Programming Languages", "Software Skills", "Soft Skills", "Language Skills",
    "Professional Skills", "Transferable Skills", "Work Experience", "Professional Experience","Work Experiences",
    "Employment History", "Internship Experience", "Volunteer Experience", "Leadership Experience",
    "Research Experience", "Teaching Experience", "Volunteer & Achievements", "Academic Qualification",
]

# Deining Map sections into main categories
SECTION_MAPPING = {
    "skills": ["skills", "technical skills", "computer skills", "programming languages", "software skills", "soft skills", "language skills", "professional skills", "transferable skills"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "internship experience", "volunteer experience", "leadership experience", "research experience", "teaching experience"],
    "education": ["education", "academic qualification"],
    "certifications": ["certifications", "licenses", "external courses"],
    "projects": ["projects"],
    "volunteer": ["volunteer experience", "volunteer & achievements", "volunteer", "achievements"],
}


# Text Cleaner
class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\u2013", "-").replace("–", "-")
        return text.strip()

class DataExtractor:
    def __init__(self, raw_text: str = cleaned_text):
        self.text = TextCleaner.clean_text(raw_text)
        self.doc = nlp(self.text)
        self.sections = self.split_into_sections()

    def split_into_sections(self):
        lines = self.text.split("\n")
        sections = {}
        current_section = "other"
        buffer = []

        for line in lines:
            line_clean = line.strip().lower()
            if any(sec.lower() == line_clean for sec in RESUME_SECTIONS):
                if buffer and current_section:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = line_clean
                buffer = []
            else:
                buffer.append(line)
        
        if buffer and current_section:
            sections[current_section] = "\n".join(buffer).strip()

        return sections

    def get_section(self, category):
        targets = SECTION_MAPPING.get(category, [])
        for section_title, content in self.sections.items():
            for target in targets:
                if target in section_title:
                    return content
        return ""

    def extract_skills(self):
        return self.get_section("skills")

    def extract_experience(self):
        return self.get_section("experience")

    def extract_education(self):
        return self.get_section("education")

    def extract_certifications(self):
        return self.get_section("certifications")

    def extract_projects(self):
        return self.get_section("projects")

    def extract_volunteer(self):
        return self.get_section("volunteer")

extractor = DataExtractor()
skills = extractor.extract_skills()
experience = extractor.extract_experience()
education = extractor.extract_education()
certification = extractor.extract_certifications()
projects = extractor.extract_projects()
volunteer = extractor.extract_volunteer()


print("Skills Section:\n", skills)
print("\nExperience Section:\n", experience)
print("\nEducation Section:\n", education)
print("\nCertifications Section:\n", certification)
print("\nProjects Section:\n", projects)
print("\nVolunteer Section:\n", volunteer)

#Cleaning the Text
cleaned_text = clean_special_characters(cleaned_text)
cleaned_text


### remove dates, Lemmatize and remove stop words. 
### Keep the clean version as well as the other. 

#Saving in Json

extracted_resume_data = {
    "personal_info": {
        "name": name,
        "emails": emails,
        "phones": phones,
        "places": places,
        "links": links
    },
    "resume_sections": {
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certification,
        "projects": projects,
        "volunteer": volunteer
    },
    "full_cleaned_text": cleaned_text
}

# Save it into a JSON file
with open("extracted_resume.json", "w", encoding="utf-8") as f:
    json.dump(extracted_resume_data, f, indent=4, ensure_ascii=False)

## Clean and Save Job Description
# (Replaces the deleted Text Cleaner Job Desc notebook)

clean_JobDesc = resume["job_description"]

def basic_text_cleaning(text):
    text = text.replace("–", "-").replace("–", "-")
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def lemmatize_and_clean(text, nlp):
    doc = nlp(text)
    return [token.lemma_.lower().strip() for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num]

semanticJobDesc = basic_text_cleaning(clean_JobDesc)
lemmatizedJobDesc = lemmatize_and_clean(clean_JobDesc, nlp)

jobdesc_data = {
    "semantic_job_desc": semanticJobDesc,
    "lemmatized_job_desc": lemmatizedJobDesc
}

with open("cleaned_job_description.json", "w", encoding="utf-8") as f:
    json.dump(jobdesc_data, f, indent=4, ensure_ascii=False)

print("Job description cleaned and saved to cleaned_job_description.json")
