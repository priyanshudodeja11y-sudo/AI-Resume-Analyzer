import io
import re

import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "postgresql",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "streamlit",
    "flask",
    "django",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "data analysis",
    "data visualization",
    "statistics",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "mongodb"
]


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_pdf_text(file_bytes):
    text = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    return text


# =========================================================
# EXTRACT SKILLS
# =========================================================

def extract_skills(text):
    lower_text = text.lower()
    found = []

    for skill in SKILLS:
        if skill.lower() in lower_text:
            found.append(skill)

    return sorted(set(found))


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


# =========================================================
# CALCULATE RESUME MATCH SCORE
# =========================================================

def similarity_score(resume, job):
    if not resume.strip() or not job.strip():
        return 0

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        [
            normalize(resume),
            normalize(job)
        ]
    )

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(score * 100, 2)


# =========================================================
# CALCULATE ATS SCORE
# =========================================================

def ats_score(text, skills):
    lower = text.lower()

    score = 0

    # Email
    if re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        lower
    ):
        score += 15

    # Phone number
    digits_only = re.sub(r"\D", "", text)

    if re.search(r"\d{10}", digits_only):
        score += 10

    # Resume sections
    sections = {
        "education": 10,
        "experience": 15,
        "projects": 15,
        "skills": 15
    }

    for section, points in sections.items():
        if section in lower:
            score += points

    # Skills
    if len(skills) >= 5:
        score += 20
    elif len(skills) >= 3:
        score += 10

    return min(score, 100)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("How it works")

    st.write("1. Upload a PDF resume")
    st.write("2. Paste the job description")
    st.write("3. Extract relevant skills")
    st.write("4. Calculate TF-IDF cosine similarity")
    st.write("5. Identify missing skills")
    st.write("6. Generate ATS score")
    st.write("7. Visualize skill matching")


# =========================================================
# MAIN TITLE
# =========================================================

st.title("📄 AI Resume Analyzer")

st.caption(
    "NLP-based resume and job-description matching tool"
)


# =========================================================
# USER INPUT
# =========================================================

resume_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder=(
        "Example: Looking for a Python developer with SQL, "
        "machine learning, Pandas, NumPy and Power BI skills..."
    )
)


# =========================================================
# ANALYSIS
# =========================================================

if resume_file and job_description.strip():

    try:

        # -------------------------------------------------
        # READ RESUME
        # -------------------------------------------------

        resume_bytes = resume_file.read()

        resume_text = extract_pdf_text(
            resume_bytes
        )


        # -------------------------------------------------
        # EXTRACT SKILLS
        # -------------------------------------------------

        resume_skills = extract_skills(
            resume_text
        )

        job_skills = extract_skills(
            job_description
        )


        # -------------------------------------------------
        # MATCHING SKILLS
        # -------------------------------------------------

        matching_skills = [
            skill
            for skill in job_skills
            if skill in resume_skills
        ]


        # -------------------------------------------------
        # MISSING SKILLS
        # -------------------------------------------------

        missing_skills = [
            skill
            for skill in job_skills
            if skill not in resume_skills
        ]


        # -------------------------------------------------
        # RESUME MATCH SCORE
        # -------------------------------------------------

        score = similarity_score(
            resume_text,
            job_description
        )


        # -------------------------------------------------
        # ATS SCORE
        # -------------------------------------------------

        ats = ats_score(
            resume_text,
            resume_skills
        )


        # =================================================
        # ANALYSIS RESULT
        # =================================================

        st.subheader("Analysis Result")


        # -------------------------------------------------
        # FOUR METRICS
        # -------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Resume Match",
            f"{score}%"
        )


        c2.metric(
            "Skills Found",
            len(resume_skills)
        )


        c3.metric(
            "Missing Job Skills",
            len(missing_skills)
        )


        c4.metric(
            "ATS Score",
            f"{ats}/100"
        )


        # -------------------------------------------------
        # MATCH PROGRESS
        # -------------------------------------------------

        st.progress(
            min(score / 100, 1.0)
        )


        # =================================================
        # SKILLS
        # =================================================

        left, right = st.columns(2)


        # -------------------------------------------------
        # LEFT COLUMN
        # -------------------------------------------------

        with left:

            st.markdown(
                "### ✅ Skills Detected"
            )

            st.write(
                ", ".join(resume_skills)
                if resume_skills
                else "No predefined skills detected."
            )


            st.markdown(
                "### 🎯 Matching Job Skills"
            )

            st.write(
                ", ".join(matching_skills)
                if matching_skills
                else "No matching job skills found."
            )


        # -------------------------------------------------
        # RIGHT COLUMN
        # -------------------------------------------------

        with right:

            st.markdown(
                "### ⚠️ Skills to Improve"
            )

            st.write(
                ", ".join(missing_skills)
                if missing_skills
                else "No major predefined skill gaps detected."
            )


        # =================================================
        # GRAPHICAL SKILL MATCH
        # =================================================

        st.markdown(
            "### 📊 Job Skill Match Visualization"
        )


        if job_skills:

            skill_chart_data = {
                "Skill": job_skills,
                "Match": [
                    1 if skill in resume_skills else 0
                    for skill in job_skills
                ]
            }


            # -------------------------------------------------
            # HORIZONTAL BAR CHART
            # -------------------------------------------------

            st.bar_chart(
                skill_chart_data,
                x="Skill",
                y="Match",
                horizontal=True
            )


            st.caption(
                "1 = Skill found in resume  |  "
                "0 = Skill missing from resume"
            )


        else:

            st.info(
                "No predefined skills were detected "
                "in the job description."
            )


        # =================================================
        # EXTRACTED RESUME TEXT
        # =================================================

        st.markdown(
            "### 📄 Extracted Resume Text"
        )


        st.text_area(
            "Text",
            resume_text[:8000],
            height=250
        )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.info(
            "Note: This is an academic prototype. "
            "The matching score is based on text similarity "
            "and should not be treated as a real hiring decision."
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            f"Could not process the PDF: {e}"
        )


else:

    st.info(
        "Upload a resume PDF and paste a job description to begin."
    )
