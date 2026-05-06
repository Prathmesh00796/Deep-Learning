import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jobs = pd.read_csv("jobs.csv")

def match_jobs(user_skills):
    job_docs = jobs["skills"].tolist()
    user_doc = " ".join(user_skills)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(job_docs + [user_doc])

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    scores = similarity.flatten()

    jobs["score"] = scores
    return jobs.sort_values(by="score", ascending=False)

def generate_advice(skills, role):
    role_skills = jobs[jobs["role"] == role]["skills"].values[0].split()

    missing = [s for s in role_skills if s not in skills]

    return f"""
🤖 Based on your profile, you are best suited for **{role}**

🧠 Your Skills:
{', '.join(skills)}

❗ Missing Skills:
{', '.join(missing[:5])}

📈 Roadmap:
1. Learn missing skills
2. Build real-world projects
3. Practice DSA (very important)
4. Apply to internships

🔥 Pro Tip:
Focus on consistency + projects over theory
"""