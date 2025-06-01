
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import docx2txt
from PyPDF2 import PdfReader
import feedparser

@st.cache_data
def load_data():
    return pd.read_csv("real_job_obsolescence_dataset.csv")

df = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def assess_risk(row):
    return 1 if row['projected_growth'] < 0 and row['automatable'] > 0.5 else 0

df['obsolete_risk'] = df.apply(assess_risk, axis=1)
df['embedding'] = df['job_title'].apply(lambda x: model.encode(x, convert_to_tensor=False).tolist())

def predict_job(job_input):
    input_emb = model.encode(job_input, convert_to_tensor=False)
    scores = [util.pytorch_cos_sim(input_emb, emb).item() for emb in df['embedding']]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    match = df.iloc[best_idx]
    risk = "⚠️ High Risk" if match['obsolete_risk'] == 1 else "✅ Low Risk"
    return match['job_title'], match['description'], risk, match['skills_required']

def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        elif file.name.endswith(".docx"):
            text = docx2txt.process(file)
            return text.strip()
        return ""
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def get_live_jobs(query="machine learning", location="INDIA"):
    try:
        feed = feedparser.parse(f"https://www.indeed.com/rss?q={query}&l={location}")
        return [{"title": entry.title, "link": entry.link} for entry in feed.entries[:5]]
    except Exception as e:
        st.warning(f"Could not fetch live jobs: {e}")
        return []

skill_suggestions = {
    "AI Research Scientist": ["Coursera: Deep Learning Specialization", "Udemy: Advanced AI Projects"],
    "Cybersecurity Analyst": ["edX: Cybersecurity Fundamentals", "Coursera: Google Cybersecurity"],
    "Software Developer": ["Coursera: Meta Front-End", "Udacity: Full Stack NanoDegree"]
}

st.title("🧠 AI Job Obsolescence Predictor")

job_input = st.text_input("Enter a job title to analyze:")

uploaded_file = st.file_uploader("Or upload a resume or job description (PDF or DOCX)", type=["pdf", "docx"])
if uploaded_file is not None:
    job_input = extract_text(uploaded_file)

if job_input:
    with st.spinner("Analyzing job..."):
        job, desc, risk, skills = predict_job(job_input)
    st.success("Prediction complete!")
    st.markdown(f"**Matched Job:** {job}")
    st.markdown(f"**Description:** {desc}")
    st.markdown(f"**Obsolescence Risk:** {risk}")
    st.markdown(f"**Recommended Skills:** {skills}")

    if job in skill_suggestions:
        st.markdown("### 📚 Courses to Build Future-Ready Skills:")
        for course in skill_suggestions[job]:
            st.markdown(f"- {course}")

    st.markdown("### 🔥 In-Demand Job Listings:")
    jobs = get_live_jobs(job_input)
    if jobs:
        for job_post in jobs:
            st.markdown(f"- [{job_post['title']}]({job_post['link']})")
    else:
        st.markdown("- No live job listings available.")

    st.markdown("### 📉 Obsolescence Risk by Job Role")
    fig, ax = plt.subplots()
    sorted_df = df.sort_values(by="automatable", ascending=False)
    ax.barh(sorted_df['job_title'], sorted_df['automatable'], color='tomato')
    ax.set_xlabel("Automation Risk")
    st.pyplot(fig)
    plt.close(fig)
