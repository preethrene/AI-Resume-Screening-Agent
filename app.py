import streamlit as st
import pandas as pd
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "orig_text" not in st.session_state:
    st.session_state.orig_text = {}
if "skills" not in st.session_state:
    st.session_state.skills = []

# ---------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------
def clean(t):
    return t.lower().strip() if isinstance(t, str) else ""

def extract_pdf_text(file):
    """Extract ORIGINAL text for better viewing later."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for p in reader.pages:
            page = p.extract_text()
            if page:
                text += page + " "
        return text
    except Exception:
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

    bigrams = list(zip(meaningful, meaningful[1:]))
    multi = [f"{w1} {w2}" for w1, w2 in bigrams if f"{w1} {w2}" in MULTI]
    singles = [w for w in meaningful if w in SINGLE]

    return list(dict.fromkeys(multi + singles))

# ---------------------------------------------------------------
# HIGHLIGHT RESUME
# ---------------------------------------------------------------
def highlight_resume(text, matched, missing):
    """Highlight matched skills + show missing skills banner."""
    if text is None:
        text = ""

    # Missing skills alert box
    missing_clean = [m for m in missing if m and m != "-"]
    banner = ""
    if missing_clean:
        banner = (
            "<div style='background:#ffe5e5;color:#b30000;padding:10px;"
            "border-left:6px solid #ff4d4d;margin-bottom:10px;font-weight:600;'>"
            "Missing Skills: " + ", ".join(missing_clean) + "</div>"
        )

    # Highlight matched skills inside resume text
    highlighted = text
    for skill in sorted(matched, key=len, reverse=True):
        pattern = rf"(?i)\b{re.escape(skill)}\b"
        replacement = (
            f"<span style='background:#d4f8d4;color:#006600;font-weight:bold;'>{skill}</span>"
        )
        highlighted = re.sub(pattern, replacement, highlighted)

    body = (
        "<div style='white-space:pre-wrap; font-family:monospace; "
        "font-size:0.9rem; line-height:1.4;'>"
        + highlighted +
        "</div>"
    )

    return banner + body

# ---------------------------------------------------------------
# COMPUTE RESULTS
# ---------------------------------------------------------------
def compute_results(jd_text, skills, df, bias):
    jd_clean = clean(jd_text)

    cleaned = []
    for _, row in df.iterrows():
        t = row["resume"]
        cleaned.append(anonymize(t, row["name"]) if bias else clean(t))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    corpus = [jd_clean] + cleaned
    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[0:1]
    res_vec = tfidf[1:]

    # TF-IDF weight of each JD skill
    skill_weights = {}
    for skill in skills:
        if skill in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[skill]
            skill_weights[skill] = float(jd_vec[0, idx])
        else:
            skill_weights[skill] = 0.0

    all_rows = []

    for i, row in df.iterrows():
        txt = cleaned[i]
        tokens = tokenize(txt)

        # Identify matched + missing skills
        matched = []
        for s in skills:
            if (" " in s and s in txt) or (s in tokens):
                matched.append(s)

        missing = [s for s in skills if s not in matched]

        score = cosine_similarity(res_vec[i:i+1], jd_vec)[0][0] * 100
        coverage = len(matched) / len(skills) * 100 if skills else 0

        all_rows.append({
            "Candidate": row["name"],
            "Score (%)": round(score,2),
            "Skill Coverage (%)": round(coverage,2),
            "Matched Skills": ", ".join(matched) if matched else "-",
            "Missing Skills": ", ".join(missing) if missing else "-",
            "TFIDF Weights": skill_weights
        })

    df_out = pd.DataFrame(all_rows).sort_values("Score (%)", ascending=False)
    df_out.insert(0, "Rank", range(1, len(df_out)+1))
    return df_out

# ---------------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------------
st.title("🤖 AI Resume Screening Agent")
st.caption("Smart JD skill extraction • TF-IDF similarity scoring • Bias-free screening")
st.divider()

# ---------------- STEP 1 --------------
st.header("📌 Step 1 — Paste Job Description")
jd_text = st.text_area("Paste JD text:", height=150)

bias = st.checkbox("Enable Bias-Free Mode")

skills = extract_jd_skills(jd_text)
st.session_state.skills = skills

if skills:
    st.success("Extracted Skills: " + ", ".join(skills))
else:
    st.warning("No valid JD skills found")

st.divider()

# ---------------- STEP 2 --------------
st.header("📁 Step 2 — Upload PDF Resumes")
files = st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True)

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
        st.error("Upload resumes first.")
    elif not skills:
        st.error("No JD skills detected.")
    else:
        with st.spinner("Processing..."):
            result = compute_results(jd_text, skills, resume_df, bias)

        st.session_state.results = result
        st.session_state.orig_text = orig_text

# ---------------- SHOW RESULTS ----------------
if st.session_state.results is not None:

    result = st.session_state.results
    orig = st.session_state.orig_text
    skills = st.session_state.skills

    st.success("Screening Completed!")
    st.dataframe(result, use_container_width=True)

    st.download_button("Download CSV", result.to_csv(index=False), "results.csv")

    st.divider()

    # ---------------- RESUME VIEWER ----------------
    st.subheader("📝 Resume Viewer (Highlighted)")

    candidate = st.selectbox("Select candidate", result["Candidate"], key="resume_view")

    row = result[result["Candidate"] == candidate].iloc[0]

    matched = [] if row["Matched Skills"]=="-" else row["Matched Skills"].split(", ")
    missing = [] if row["Missing Skills"]=="-" else row["Missing Skills"].split(", ")

    html_resume = highlight_resume(orig.get(candidate,""), matched, missing)
    st.markdown(html_resume, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # 📘 ATS EXPLANATION
    # ---------------------------------------------------------------
    st.subheader("📘 ATS Keyword Score Explanation")

    weights = row["TFIDF Weights"]

    exp_df = pd.DataFrame({
        "Skill": skills,
        "TF-IDF Weight": [round(weights[s],4) for s in skills],
        "Status": ["Matched" if s in matched else "Missing" for s in skills]
    })

    st.dataframe(exp_df, use_container_width=True)

    explanation = f"""
    **Why {candidate} scored {row['Score (%)']}%?**

    - TF-IDF score is based on keyword importance from the JD.
    - The candidate matched **{len(matched)} / {len(skills)}** JD skills.
    - Missing: {', '.join(missing) if missing else 'None'}
    - High-weight JD skills contributing strongly:
      **{', '.join([s for s in skills if weights[s] > 0.1]) or 'None'}**
    """

    st.markdown(explanation)

    # -------------------------------------------------------
    # 📊 DASHBOARD
    # -------------------------------------------------------
    st.divider()
    st.header("📊 Candidate Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Resumes Processed", len(result))
    with col2:
        st.metric("🧠 JD Skills Found", len(skills))
    with col3:
        st.metric("🥇 Highest Score", f"{result['Score (%)'].max()}%")
    with col4:
        st.metric("📊 Average Score", f"{round(result['Score (%)'].mean(), 2)}%")

    st.divider()

    fig_score = px.bar(
        result,
        x="Candidate",
        y="Score (%)",
        title="Candidate Match Score",
        text="Score (%)",
        color="Score (%)",
        color_continuous_scale="Blues"
    )
    fig_score.update_traces(textposition="outside")
    st.plotly_chart(fig_score, use_container_width=True)

    fig_cov = px.bar(
        result,
        x="Candidate",
        y="Skill Coverage (%)",
        title="Skill Coverage (%)",
        text="Skill Coverage (%)",
        color="Skill Coverage (%)",
        color_continuous_scale="Greens"
    )
    fig_cov.update_traces(textposition="outside")
    st.plotly_chart(fig_cov, use_container_width=True)

    st.divider()

    st.subheader("🧩 Matched vs Missing Skills")
    selected_person = st.selectbox("Select candidate", result["Candidate"], key="pie")

    rp = result[result["Candidate"] == selected_person].iloc[0]

    m_matched = [] if rp["Matched Skills"]=="-" else rp["Matched Skills"].split(", ")
    m_missing = [] if rp["Missing Skills"]=="-" else rp["Missing Skills"].split(", ")

    fig_pie = px.pie(
        names=["Matched","Missing"],
        values=[len(m_matched), len(m_missing)],
        color=["Matched","Missing"],
        color_discrete_map={"Matched":"green","Missing":"red"},
        title=f"Skill Distribution for {selected_person}"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
