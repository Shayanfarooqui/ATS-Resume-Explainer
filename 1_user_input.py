import json

UserInputResume = ''' Experienced Data Analyst with a strong background in SQL, Python, and data visualization. Skilled in cleaning and analyzing complex datasets, building interactive dashboards, and presenting actionable insights to stakeholders. Passionate about turning data into strategic decisions.

Skills

SQL, Python (pandas, NumPy)

Tableau, Power BI

Data cleaning & wrangling

Exploratory Data Analysis (EDA)

Dashboard design and reporting

Statistical analysis

Professional Experience

Data Analyst
XYZ Corporation – New York, NY
Jan 2022 – Present

Performed data cleaning and transformation using Python (pandas, NumPy) to prepare large datasets for analysis.

Wrote complex SQL queries to extract and manipulate data from relational databases.

Built interactive dashboards using Tableau to visualize KPIs and business metrics.

Conducted exploratory data analysis to uncover trends in customer behavior.

Presented findings and strategic insights to business stakeholders using visual reports.

Collaborated with product and marketing teams to define KPIs and monitor campaign performance using Power BI.

Education
Bachelor of Science in Data Science
XYZ University – 2020

Certifications

Google Data Analytics Certificate

Tableau Desktop Specialist

Projects

Customer Churn Prediction: Built a classification model in scikit-learn with 85% accuracy.

Sales Dashboard: Developed a real-time Power BI dashboard for a retail client.

Volunteer Work

Data Visualization Mentor – Local coding bootcamp (2023)
'''

UserInputJobDesc = '''
Location: New York, NY (Hybrid)
Company: XYZ Corporation
Type: Full-time

Job Description
XYZ Corporation is seeking a detail-oriented Data Analyst to join our Business Intelligence team. You will work with large datasets to uncover insights, support decision-making, and build interactive dashboards that drive business performance.

Responsibilities

Clean, organize, and analyze structured and unstructured data from multiple sources

Write advanced SQL queries to extract actionable insights

Perform exploratory data analysis (EDA) to identify trends and anomalies

Build interactive dashboards and visualizations using Tableau or Power BI

Present findings to business leaders and make data-driven recommendations

Collaborate with cross-functional teams (marketing, product, finance) to define KPIs

Monitor ongoing business performance and optimize reports accordingly

Requirements

1-3 years of experience in a Data Analyst or BI role

Strong skills in SQL and Python (pandas, NumPy)

Experience with data visualization tools like Tableau, Power BI

Understanding of statistical methods and A/B testing

Excellent communication and presentation skills

Bachelor's degree in Data Science, Statistics, or a related field

Preferred

Experience with predictive modeling or machine learning tools

Familiarity with Google Analytics or other web analytics platforms

Certification in Tableau, Power BI, or Google Data Analytics

Why Join Us?

Competitive salary and benefits

Work with a fast-growing, data-driven team

Opportunity to make a measurable impact on company strategy
'''

user_inputs = {
    "resume": UserInputResume,
    "job_description": UserInputJobDesc
}

with open("user_inputs.json", "w", encoding="utf-8") as f:
    json.dump(user_inputs, f, indent=4, ensure_ascii=False)

print("Saved user_inputs.json")
