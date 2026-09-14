import io
import re

import pdfplumber
import pandas as pd
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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9aa0a6;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
    }

    .skill-box {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px;
    }

    .small-text {
        color: #9aa0a6;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    "mongodb",
    "matplotlib",
    "seaborn",
    "powerpoint",
    "rest api",
    "api",
    "linux",
    "spark",
    "hadoop",
    "computer vision",
    "opencv",
    "scipy",
    "r",
    "c",
    "c#",
    "typescript"
]


# =========================================================
# COMMON RESUME SECTIONS
# =========================================================

RESUME_SECTIONS = {
    "education": "Education",
    "experience": "Experience",
    "projects": "Projects",
    "skills": "Skills",
    "certifications": "Certifications",
    "summary": "Summary",
    "objective": "Objective",
    "achievements": "Achievements"
}


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

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, lower_text):

            found.append(skill)

    return sorted(set(found))


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


# =========================================================
# RESUME MATCH SCORE
# =========================================================

def similarity_score(resume, job):

    if not resume.strip() or not job.strip():

        return 0

    try:

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

    except Exception:

        return 0


# =========================================================
# ATS SCORE
# =========================================================

def ats_score(text, skills):

    lower = text.lower()

    score = 0

    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------

    if re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text
    ):

        score += 15


    # -----------------------------------------------------
    # Phone
    # -----------------------------------------------------

    digits_only = re.sub(
        r"\D",
        "",
        text
    )

    if re.search(
        r"\d{10}",
        digits_only
    ):

        score += 10


    # -----------------------------------------------------
    # Resume sections
    # -----------------------------------------------------

    sections = {
        "education": 10,
        "experience": 15,
        "projects": 15,
        "skills": 15
    }

    for section, points in sections.items():

        if section in lower:

            score += points


    # -----------------------------------------------------
    # Skills
    # -----------------------------------------------------

    if len(skills) >= 5:

        score += 20

    elif len(skills) >= 3:

        score += 10

    return min(score, 100)


# =========================================================
# RESUME SECTION CHECK
# =========================================================

def check_sections(text):

    lower = text.lower()

    found = []
    missing = []

    for key, name in RESUME_SECTIONS.items():

        if key in lower:

            found.append(name)

        else:

            missing.append(name)

    return found, missing


# =========================================================
# RESUME STATISTICS
# =========================================================

def resume_statistics(text):

    words = re.findall(
        r"\b\w+\b",
        text
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return {
        "words": len(words),
        "characters": len(text),
        "lines": len(lines)
    }


# =========================================================
# MATCH PERCENTAGE
# =========================================================

def skill_match_percentage(job_skills, matching_skills):

    if not job_skills:

        return 0

    return round(
        (len(matching_skills) / len(job_skills)) * 100,
        2
    )


# =========================================================
# SCORE INTERPRETATION
# =========================================================

def score_message(score):

    if score >= 80:

        return (
            "Excellent match. Your resume contains strong "
            "alignment with the job description."
        )

    elif score >= 60:

        return (
            "Good match. Your resume is reasonably aligned, "
            "but some improvements could increase the match."
        )

    elif score >= 40:

        return (
            "Moderate match. Several relevant keywords or "
            "skills are missing from the resume."
        )

    else:

        return (
            "Low match. The resume and job description have "
            "limited textual similarity."
        )


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    missing_skills,
    missing_sections,
    ats,
    score
):

    recommendations = []

    # Missing skills
    if missing_skills:

        top_skills = missing_skills[:5]

        recommendations.append(
            "Consider adding relevant skills such as "
            + ", ".join(top_skills)
            + " if you genuinely possess them."
        )


    # Missing sections
    if "Projects" in missing_sections:

        recommendations.append(
            "Add a Projects section with 2–3 relevant "
            "projects and measurable outcomes."
        )


    if "Experience" in missing_sections:

        recommendations.append(
            "If applicable, add internships, work experience, "
            "or practical training."
        )


    if "Certifications" in missing_sections:

        recommendations.append(
            "Add relevant certifications if you have completed "
            "any industry-recognized courses."
        )


    # ATS
    if ats < 70:

        recommendations.append(
            "Improve ATS readiness by using clear section "
            "headings and including relevant job keywords."
        )


    # Similarity
    if score < 50:

        recommendations.append(
            "Rewrite parts of the resume using terminology "
            "that accurately reflects the target job description."
        )


    if not recommendations:

        recommendations.append(
            "Your resume is well aligned. Focus on measurable "
            "achievements and keeping the content concise."
        )


    return recommendations


# =========================================================
# CREATE DOWNLOAD REPORT
# =========================================================

def create_report(
    score,
    ats,
    resume_skills,
    job_skills,
    matching_skills,
    missing_skills,
    statistics
):

    report = []

    report.append(
        "AI RESUME ANALYZER - ANALYSIS REPORT"
    )

    report.append("=" * 45)

    report.append(
        f"Resume Match Score: {score}%"
    )

    report.append(
        f"ATS Score: {ats}/100"
    )

    report.append(
        f"Skill Match: "
        f"{skill_match_percentage(job_skills, matching_skills)}%"
    )

    report.append("")

    report.append(
        "RESUME SKILLS"
    )

    report.append(
        ", ".join(resume_skills)
        if resume_skills
        else "None detected"
    )

    report.append("")

    report.append(
        "JOB REQUIRED SKILLS"
    )

    report.append(
        ", ".join(job_skills)
        if job_skills
        else "None detected"
    )

    report.append("")

    report.append(
        "MATCHING SKILLS"
    )

    report.append(
        ", ".join(matching_skills)
        if matching_skills
        else "None"
    )

    report.append("")

    report.append(
        "MISSING SKILLS"
    )

    report.append(
        ", ".join(missing_skills)
        if missing_skills
        else "None"
    )

    report.append("")

    report.append(
        "RESUME STATISTICS"
    )

    report.append(
        f"Words: {statistics['words']}"
    )

    report.append(
        f"Characters: {statistics['characters']}"
    )

    report.append(
        f"Text Lines: {statistics['lines']}"
    )

    report.append("")

    report.append(
        score_message(score)
    )

    return "\n".join(report)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 How It Works")

    st.write(
        "1. Upload a PDF resume"
    )

    st.write(
        "2. Paste the job description"
    )

    st.write(
        "3. Extract resume skills"
    )

    st.write(
        "4. Extract job skills"
    )

    st.write(
        "5. Compare both texts"
    )

    st.write(
        "6. Calculate ATS score"
    )

    st.write(
        "7. Identify missing skills"
    )

    st.write(
        "8. Visualize the results"
    )

    st.divider()

    st.caption(
        "Academic prototype • NLP + TF-IDF"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP-based resume and job-description matching tool'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Upload Resume")

resume_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)


st.subheader("Job Description")

job_description = st.text_area(
    "Paste the Job Description",
    height=220,
    placeholder=(
        "Example: Looking for a Python developer with "
        "SQL, machine learning, Pandas, NumPy, Power BI "
        "and data analysis skills..."
    )
)


# =========================================================
# ANALYSIS BUTTON
# =========================================================

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    if not resume_file:

        st.error(
            "Please upload a PDF resume first."
        )

        st.stop()


    if not job_description.strip():

        st.error(
            "Please paste a job description first."
        )

        st.stop()


    try:

        with st.spinner(
            "Analyzing resume..."
        ):

            # -------------------------------------------------
            # EXTRACT RESUME TEXT
            # -------------------------------------------------

            resume_bytes = resume_file.read()

            resume_text = extract_pdf_text(
                resume_bytes
            )


            if not resume_text.strip():

                st.error(
                    "No readable text was found in the PDF. "
                    "Please upload a text-based PDF."
                )

                st.stop()


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
            # SCORES
            # -------------------------------------------------

            score = similarity_score(
                resume_text,
                job_description
            )

            ats = ats_score(
                resume_text,
                resume_skills
            )

            skill_percentage = skill_match_percentage(
                job_skills,
                matching_skills
            )


            # -------------------------------------------------
            # RESUME SECTIONS
            # -------------------------------------------------

            found_sections, missing_sections = check_sections(
                resume_text
            )


            # -------------------------------------------------
            # RESUME STATISTICS
            # -------------------------------------------------

            statistics = resume_statistics(
                resume_text
            )


            # -------------------------------------------------
            # RECOMMENDATIONS
            # -------------------------------------------------

            recommendations = generate_recommendations(
                missing_skills,
                missing_sections,
                ats,
                score
            )


        # =====================================================
        # RESULT HEADER
        # =====================================================

        st.divider()

        st.header(
            "📊 Analysis Result"
        )


        # =====================================================
        # SCORE CARDS
        # =====================================================

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Resume Match",
            f"{score}%"
        )


        c2.metric(
            "ATS Score",
            f"{ats}/100"
        )


        c3.metric(
            "Skill Match",
            f"{skill_percentage}%"
        )


        c4.metric(
            "Missing Skills",
            len(missing_skills)
        )


        # =====================================================
        # SCORE EXPLANATION
        # =====================================================

        if score >= 80:

            st.success(
                "🟢 " + score_message(score)
            )

        elif score >= 50:

            st.warning(
                "🟡 " + score_message(score)
            )

        else:

            st.error(
                "🔴 " + score_message(score)
            )


        st.progress(
            min(score / 100, 1.0)
        )


        # =====================================================
        # SKILLS SECTION
        # =====================================================

        st.header(
            "🧠 Skill Analysis"
        )


        left, right = st.columns(2)


        # -----------------------------------------------------
        # DETECTED SKILLS
        # -----------------------------------------------------

        with left:

            st.subheader(
                "✅ Skills Detected"
            )

            if resume_skills:

                st.write(
                    ", ".join(resume_skills)
                )

            else:

                st.info(
                    "No predefined skills detected."
                )


            st.subheader(
                "🎯 Matching Job Skills"
            )

            if matching_skills:

                st.write(
                    ", ".join(matching_skills)
                )

            else:

                st.info(
                    "No matching job skills found."
                )


        # -----------------------------------------------------
        # MISSING SKILLS
        # -----------------------------------------------------

        with right:

            st.subheader(
                "⚠️ Skills to Improve"
            )

            if missing_skills:

                st.write(
                    ", ".join(missing_skills)
                )

            else:

                st.success(
                    "No major predefined skill gaps detected."
                )


            st.subheader(
                "💼 Skills Required by Job"
            )

            if job_skills:

                st.write(
                    ", ".join(job_skills)
                )

            else:

                st.info(
                    "No predefined skills detected in job description."
                )


        # =====================================================
        # GRAPH
        # =====================================================

        st.header(
            "📊 Job Skill Match Visualization"
        )


        if job_skills:

            chart_data = pd.DataFrame(
                {
                    "Skill": job_skills,
                    "Match": [
                        1 if skill in resume_skills else 0
                        for skill in job_skills
                    ]
                }
            )


            chart_data = chart_data.set_index(
                "Skill"
            )


            st.bar_chart(
                chart_data,
                horizontal=True
            )


            st.caption(
                "1 = Skill found in resume  |  "
                "0 = Skill missing from resume"
            )


        else:

            st.info(
                "The job description did not contain "
                "any predefined skills."
            )


        # =====================================================
        # RESUME STATISTICS
        # =====================================================

        st.header(
            "📈 Resume Statistics"
        )


        s1, s2, s3, s4 = st.columns(4)


        s1.metric(
            "Words",
            statistics["words"]
        )


        s2.metric(
            "Characters",
            statistics["characters"]
        )


        s3.metric(
            "Text Lines",
            statistics["lines"]
        )


        s4.metric(
            "Skills Detected",
            len(resume_skills)
        )


        # =====================================================
        # RESUME SECTIONS
        # =====================================================

        st.header(
            "📋 Resume Structure"
        )


        section_left, section_right = st.columns(2)


        with section_left:

            st.subheader(
                "✅ Sections Detected"
            )

            if found_sections:

                for section in found_sections:

                    st.write(
                        f"• {section}"
                    )

            else:

                st.write(
                    "No standard sections detected."
                )


        with section_right:

            st.subheader(
                "⚠️ Sections Not Detected"
            )

            if missing_sections:

                for section in missing_sections:

                    st.write(
                        f"• {section}"
                    )

            else:

                st.success(
                    "All standard sections detected."
                )


        # =====================================================
        # IMPROVEMENT RECOMMENDATIONS
        # =====================================================

        st.header(
            "💡 Recommendations"
        )


        for recommendation in recommendations:

            st.write(
                f"🔹 {recommendation}"
            )


        # =====================================================
        # EXTRACTED RESUME TEXT
        # =====================================================

        st.header(
            "📄 Extracted Resume Text"
        )


        st.text_area(
            "Resume Content",
            resume_text[:12000],
            height=300
        )


        # =====================================================
        # DOWNLOAD REPORT
        # =====================================================

        report = create_report(
            score,
            ats,
            resume_skills,
            job_skills,
            matching_skills,
            missing_skills,
            statistics
        )


        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )


        # =====================================================
        # DISCLAIMER
        # =====================================================

        st.divider()

        st.info(
            "Note: This is an academic prototype. "
            "The matching score is based on TF-IDF text similarity "
            "and predefined skill detection. It should not be "
            "treated as a real hiring decision."
        )


    except Exception as e:

        st.error(
            "Something went wrong while analyzing the resume."
        )

        st.code(
            str(e)
        )


# =========================================================
# INITIAL STATE
# =========================================================

else:

    st.info(
        "Upload a resume PDF, paste a job description, "
        "and click 'Analyze Resume' to begin."
    )
