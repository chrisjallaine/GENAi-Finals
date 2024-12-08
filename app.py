import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import openai

# Set the OpenAI API key directly in the code
openai.api_key = "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"  # Replace this with your actual OpenAI API key

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Function to summarize text using the new OpenAI API
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text: {text[:1500]}"}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        summary = response['choices'][0]['message']['content'].strip()  # Fixed line
        return summary
    except Exception as e:
        return f"Error in summarization: {e}"

# Function to store data in SQLite3
def store_data_in_db(job_description, resumes, scores, summaries):
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resume_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_description TEXT,
                    resume_name TEXT,
                    resume_text TEXT,
                    score REAL,
                    summary TEXT)''')
    for i, resume_text in enumerate(resumes):
        c.execute('''INSERT INTO resume_data (job_description, resume_name, resume_text, score, summary)
                     VALUES (?, ?, ?, ?, ?)''',
                  (job_description, uploaded_files[i].name, resume_text, scores[i], summaries[i]))
    conn.commit()
    conn.close()

# Streamlit app
st.title("HR Resume Screening Assistance Tool")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Results")
    
    resumes = []
    summaries = []
    
    # Extract text and summarize each resume
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        summary = summarize_text(text)  # Summarize the resume text
        summaries.append(summary)
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    
    # Store data in SQLite3
    store_data_in_db(job_description, resumes, scores, summaries)
    
    # Display results
    results = pd.DataFrame({
        "Resume Name": [file.name for file in uploaded_files],
        "Score": scores,
        "Summary": summaries
    })
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
    
    # Highlight top matches
    st.subheader("Top Match")
    top_match_index = results["Score"].idxmax()
    st.write(f"Top Resume: **{results.iloc[top_match_index]['Resume Name']}**")
    st.write(f"Score: **{results.iloc[top_match_index]['Score']:.2f}**")
    st.write(f"Summary: {results.iloc[top_match_index]['Summary']}")