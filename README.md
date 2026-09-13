# AI Resume Analyzer

An academic mini-project that uses Natural Language Processing (NLP) to compare a resume with a job description.

## Features
- PDF resume text extraction
- Skill extraction using a predefined skill dictionary
- TF-IDF based text representation
- Cosine similarity based match score
- Missing-skill identification
- Streamlit web interface

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Method
1. Extract text from the uploaded PDF using pdfplumber.
2. Normalize the text.
3. Detect predefined technical skills.
4. Convert resume and job description into TF-IDF vectors.
5. Calculate cosine similarity.
6. Compare detected skills to identify gaps.

## Limitations
- The skill list is predefined.
- PDF extraction quality depends on resume formatting.
- Similarity score is not a complete measure of candidate suitability.
- It does not understand experience quality or context like a production-grade recruitment system.
