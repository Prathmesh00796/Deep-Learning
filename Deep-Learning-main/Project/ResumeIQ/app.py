import streamlit as st
import os
from dotenv import load_dotenv

from utils.ingestion import parse_uploaded_file
from utils.chains import get_ats_score, get_job_match, chat_with_resume, setup_chat_memory

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ResumeIQ", page_icon="📄", layout="wide")

st.title("ResumeIQ 📄 - Intelligent Resume Analysis")

# Sidebar for Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API Key", value=env_api_key, type="password", help="Alternatively, set GROQ_API_KEY in the .env file")
    st.markdown("---")
    st.info("Upload your resume and enter your API key to get started.")

# Initialize session state for memory and resume
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "memory" not in st.session_state:
    st.session_state.memory = setup_chat_memory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tabs = st.tabs(["📤 Upload & ATS Score", "🎯 Job Match", "💬 Chat with Resume"])

# Tab 1: Upload & ATS Score
with tabs[0]:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    
    if st.button("Analyze Resume"):
        if not api_key:
            st.error("Please provide a Groq API Key in the sidebar.")
        elif not uploaded_file:
            st.warning("Please upload a resume first.")
        else:
            with st.spinner("Parsing format and generating embeddings..."):
                try:
                    # 1. Parse file
                    documents = parse_uploaded_file(uploaded_file)
                    resume_text = "\n".join([doc.page_content for doc in documents])
                    st.session_state.resume_text = resume_text
                    
                    # 2. ATS Score
                    with st.spinner("Calculating ATS Score..."):
                        ats_result = get_ats_score(resume_text, api_key)
                        st.markdown(ats_result)
                        st.success("Analysis Complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Tab 2: Job Match
with tabs[1]:
    st.header("Job Description Match")
    st.write("Compare your resume against a specific job description.")
    job_description = st.text_area("Paste Job Description Here", height=250)
    
    if st.button("Calculate Match"):
        if not api_key:
            st.error("Please provide a Groq API Key.")
        elif not st.session_state.resume_text:
            st.warning("Please upload and analyze your resume first in the 'Upload' tab.")
        elif not job_description:
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Analyzing match..."):
                try:
                    match_result = get_job_match(st.session_state.resume_text, job_description, api_key)
                    st.markdown(match_result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 3: Chat with Resume
with tabs[2]:
    st.header("Interactive Resume Chat")
    st.write("Ask questions about your resume based on what you uploaded.")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
    user_query = st.chat_input("Ask a question (e.g., 'What skills am I missing for a Python developer role?')")
    
    if user_query:
        if not api_key:
            st.error("Please provide a Groq API Key.")
        elif not st.session_state.resume_text:
            st.warning("Please upload and analyze a resume first.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chat_with_resume(user_query, st.session_state.resume_text, st.session_state.memory, api_key)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
