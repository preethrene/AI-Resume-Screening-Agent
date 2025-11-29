import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------
# CLEAN UTILS
# ---------------------------------------------------------------
def clean(t):
    return t.lower().strip() if isinstance(t, str) else ""

def extract_pdf_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for p in reader.pages:
            page = p.extract_text()
            if page:
                text += page + " "
        return clean(text)
    except:
        return ""

def tokenize(t):
    return set(re.findall(r"[a-z0-9#]+", t.lower()))

def anonymize(txt, name):
    if name:
        txt = re.sub(re.escape(name), " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\+?\d[\d\- ]{7,}\d", " ", txt)
    return clean(txt)

# ---------------------------------------------------------------
# JD SKILL EXTRACTION
# ---------------------------------------------------------------
STOP = {"and","or","with","for","the","a","an","in","on","of","to","is","as","are"}
BAD = {"hiring","looking","seeking","join","job","role","team",
       "responsibilities","requirements","requirement","experience"}

MULTI = {
    "machine learning","deep learning","data science","data analysis",
    "data analytics","data visualization","artificial intelligence"
}

SINGLE = {
    "python","sql","java","c++","pandas","numpy","ml","ai",
    "excel","tableau","powerbi","nlp","pytorch","tensorflow"
}

def extract_jd_skills(text):
    text = clean(text)
    words = re.findall(r"[a-zA-Z0-9#]+", text)
    meaningful = [w for w in words if w not in STOP and w not in BAD]

    # Bigrams
    bigrams = list(zip(meaningful, meaningful[1:]))
    multi = []
    for w1, w2 in bigrams:
        phrase = f"{w1} {w2}"
        if phrase in MULTI:
            multi.append(phrase)

    singles = [w for w in meaningful if w in SINGLE]

    # keep order, remove duplicates
    return list(dict.fromkeys(multi + singles))

# ---------------------------------------------------------------
# HIGHLIGHT RESUMES
# ---------------------------------------------------------------
def highlight(text, matched, missing):
    text = clean(text)
    for skill in sorted(matched + missing, key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b{re.escape(skill)}\b",
            f"**[{skill}]**",
            text
        )
    return text

# ---------------------------------------------------------------
# COMPUTE RESULTS
# ---------------------------------------------------------------
def compute_results(jd_text, skills, df, bias):
    jd_clean = clean(jd_text)

    cleaned = []
    for _, row in df.iterrows():
        t = row["resume"]
        if bias:
            t = anonymize(t, row["name"])
        cleaned.append(clean(t))

    corpus = [jd_clean] + cleaned

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[0:1]
    res_vec = tfidf[1:]

    all_rows = []

    for i, row in df.iterrows():
        txt = cleaned[i]
        tokens = tokenize(txt)

        # matching
        matched = []
        for s in skills:
            if " " in s:
                if s in txt:
                    matched.append(s)
            else:
                if s in tokens:
                    matched.append(s)

        missing = [s for s in skills if s not in matched]

        score = cosine_similarity(res_vec[i:i+1], jd_vec)[0][0] * 100
        coverage = (len(matched) / len(skills) * 100) if skills else 0

        all_rows.append({
            "Candidate": row["name"],
            "Score (%)": round(score,2),
            "Skill Coverage (%)": round(coverage,2),
            "Matched Skills": ", ".join(matched) if matched else "-",
            "Missing Skills": ", ".join(missing) if missing else "-"
        })

    df_out = pd.DataFrame(all_rows).sort_values("Score (%)", ascending=False)
    df_out.insert(0, "Rank", range(1, len(df_out)+1))
    return df_out

# ---------------------------------------------------------------
# UI LAYOUT — PURE STREAMLIT (NO CSS)
# ---------------------------------------------------------------

st.title("🤖 AI Resume Screening Agent")
st.caption("Smart JD skill extraction • TF-IDF similarity scoring • Bias-free screening")

st.divider()

# ---------------- STEP 1 --------------
st.header("📌 Step 1 — Paste Job Description")
jd_text = st.text_area("Paste JD text here:", height=150)

bias = st.checkbox("Enable Bias-Free Mode (remove names/emails/phones)")

skills = extract_jd_skills(jd_text)

if skills:
    st.success(f"Extracted Skills: {', '.join(skills)}")
else:
    st.warning("No valid JD skills detected.")

st.divider()

# ---------------- STEP 2 --------------
st.header("📁 Step 2 — Upload PDF Resumes")
files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

resume_df = None
orig_text = {}

if files:
    rows = []
    for f in files:
        txt = extract_pdf_text(f)
        name = f.name.replace(".pdf","")
        rows.append({"name": name, "resume": txt})
        orig_text[name] = txt
    resume_df = pd.DataFrame(rows)
    st.success(f"{len(files)} resumes uploaded.")

st.divider()

# ---------------- STEP 3 --------------
st.header("🚀 Step 3 — Analyze & Rank Candidates")

if st.button("Run Screening"):
    if resume_df is None:
        st.error("Please upload resumes first.")
    elif not skills:
        st.error("No JD skills detected.")
    else:
        with st.spinner("Processing..."):
            result = compute_results(jd_text, skills, resume_df, bias)

        st.success("Screening Completed!")
        st.dataframe(result, use_container_width=True)

        st.download_button("Download CSV", result.to_csv(index=False), "results.csv")

        st.divider()

        st.subheader("📝 Resume Viewer (Highlighted)")
        name = st.selectbox("Select candidate", result["Candidate"])
        row = result[result["Candidate"] == name].iloc[0]

        matched = [] if row["Matched Skills"]=="-" else row["Matched Skills"].split(", ")
        missing = [] if row["Missing Skills"]=="-" else row["Missing Skills"].split(", ")

        st.text_area(
            "Highlighted Resume",
            highlight(orig_text[name], matched, missing),
            height=350
        )
