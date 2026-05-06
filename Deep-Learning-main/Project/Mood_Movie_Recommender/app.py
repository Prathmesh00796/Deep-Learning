import streamlit as st
import pdfplumber
from utils import match_jobs, generate_advice
from github_analyzer import get_github_skills

# ---------- Resume Parser ----------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

skills_db = [
    "python","java","sql","machine learning","deep learning",
    "nlp","tensorflow","pytorch","pandas","numpy",
    "html","css","javascript","react","nodejs",
    "n8n","agenticAI","genAI","LLMs"
]

def extract_skills(text):
    return [s for s in skills_db if s in text]

# ---------- UI ----------
st.set_page_config(page_title="AI Career Advisor", layout="centered")

st.title("🤖 AI Career Advisor (Advanced)")

mode = st.radio("Choose Mode:", ["Resume", "Manual", "GitHub"])

user_skills = []

# -------- Resume --------
if mode == "Resume":
    file = st.file_uploader("Upload Resume", type=["pdf"])
    if file:
        text = extract_text(file)
        user_skills = extract_skills(text)

# -------- Manual --------
elif mode == "Manual":
    skills_input = st.text_input("Enter skills (comma separated)")
    if skills_input:
        user_skills = [s.strip().lower() for s in skills_input.split(",")]

# -------- GitHub --------
else:
    username = st.text_input("Enter GitHub Username")
    if username:
        user_skills = get_github_skills(username)

# ---------- OUTPUT ----------
if user_skills:
    st.subheader("🧠 Skills Detected")
    st.write(user_skills)

    results = match_jobs(user_skills)

    st.subheader("🏆 Top Matches")
    for _, row in results.head(3).iterrows():
        st.write(f"{row['role']} (score: {round(row['score'],2)})")

    best_role = results.iloc[0]["role"]

    st.subheader("💬 AI Career Advice")
    st.markdown(generate_advice(user_skills, best_role))

# ---------- Chatbot ----------
st.subheader("💬 Ask Career AI")

user_query = st.text_input("Ask anything (e.g. how to become AI engineer?)")

if user_query:
    st.write("🤖 AI:")
    st.write(f"""
To answer your question:

👉 {user_query}

Focus on:
- Strong fundamentals
- Real projects
- Consistency

(This is rule-based — can be upgraded with LLM)
""")
