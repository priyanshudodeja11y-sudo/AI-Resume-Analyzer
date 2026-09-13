import re
import io
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

SKILLS = [
    "python", "java", "c++", "sql", "mysql", "postgresql", "excel",
    "power bi", "tableau", "machine learning", "deep learning", "nlp",
    "natural language processing", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "keras", "streamlit", "flask", "django",
    "git", "github", "docker", "aws", "azure", "data analysis",
    "data visualization", "statistics", "html", "css", "javascript",
    "react", "node.js", "mongodb"
]

def extract_pdf_text(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def extract_skills(text):
    lower = text.lower()
    found = []
    for skill in SKILLS:
        if skill.lower() in lower:
            found.append(skill)
    return sorted(set(found))

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def similarity_score(resume, job):
    if not resume.strip() or not job.strip():
        return 0
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([normalize(resume), normalize(job)])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

st.title("📄 AI Resume Analyzer")
st.caption("NLP-based resume and job-description matching tool")

with st.sidebar:
    st.header("How it works")
    st.write("1. Upload a PDF resume")
    st.write("2. Paste the job description")
    st.write("3. Extract relevant skills")
    st.write("4. Calculate TF-IDF cosine similarity")
    st.write("5. Identify missing skills")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Example: Looking for a Python developer with SQL, machine learning and Power BI skills..."
)

if resume_file and job_description:
    try:
        resume_text = extract_pdf_text(resume_file.read())
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        missing_skills = [s for s in job_skills if s not in resume_skills]
        score = similarity_score(resume_text, job_description)

        st.subheader("Analysis Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resume Match", f"{score}%")
        c2.metric("Skills Found", len(resume_skills))
        c3.metric("Missing Job Skills", len(missing_skills))

        st.progress(min(score / 100, 1.0))

        left, right = st.columns(2)
        with left:
            st.markdown("### ✅ Skills Detected")
            st.write(", ".join(resume_skills) if resume_skills else "No predefined skills detected.")
        with right:
            st.markdown("### ⚠️ Skills to Improve")
            st.write(", ".join(missing_skills) if missing_skills else "No major predefined skill gaps detected.")

        st.markdown("### Extracted Resume Text")
        st.text_area("Text", resume_text[:8000], height=250)

        st.info(
            "Note: This is an academic prototype. The matching score is based on text similarity "
            "and should not be treated as a real hiring decision."
        )
    except Exception as e:
        st.error(f"Could not process the PDF: {e}")
else:
    st.info("Upload a resume PDF and paste a job description to begin.")
