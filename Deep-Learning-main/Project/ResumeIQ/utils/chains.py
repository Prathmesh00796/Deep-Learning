import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

def get_llm(api_key):
    # Initialize Groq LLM
    if not api_key:
        raise ValueError("Groq API Key is missing.")
    return ChatGroq(temperature=0.2, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

def get_ats_score(resume_text, api_key):
    llm = get_llm(api_key)
    prompt = PromptTemplate(
        input_variables=["resume"],
        template="""You are an expert ATS (Applicant Tracking System) and Senior Technical Recruiter.
Your goal is to provide a harsh, hyper-realistic quantitative ATS Score out of 100 based on real parsing mechanics. Do not be overly generous. Most resumes should score between 60-80.

Calculate the score rigidly based on these 5 criteria:
1. Action Verbs & Impact Metrics (0-25 pts): Are bullets outcome-driven with numbers/stats, or just passive duties?
2. Structuring & Navigability (0-20 pts): Does it have universally recognized sections (Experience, Education, Skills)?
3. Keyword Clarity (0-30 pts): Does it clearly index hard technical skills or industry tools without fluffy jargon?
4. Formatting (0-15 pts): Does the text indicate clean bullet points vs massive unreadable paragraphs?
5. Professionalism (0-10 pts).

Resume to Evaluate:
{resume}

Follow this exact Response Format:

### 🎯 Final ATS Score: [Sum of criteria]/100

### 📊 Score Breakdown:
- **Impact Metrics (Action/Results):** [X]/25
- **Structure & Sections:** [X]/20
- **Keywords/Hard Skills:** [X]/30
- **Syntax/Formatting:** [X]/15
- **Professionalism:** [X]/10

### 🔴 Critical Issues (Why you lost points):
- (List 2-4 harsh, realistic issues found in their text)

### 💡 High-Impact Recommendations:
- (List 3 specific modifications they should make to improve their score)
"""
    )
    chain = prompt | llm
    response = chain.invoke({"resume": resume_text})
    return response.content if hasattr(response, "content") else response

def get_job_match(resume_text, job_description, api_key):
    llm = get_llm(api_key)
    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""You are an expert technical recruiter analyzing a candidate against a job description.
Compare the following resume against the job description.

Resume:
{resume}

Job Description:
{jd}

Provide the following:
1. A Match Percentage (0-100%).
2. Missing Keywords/Skills that are in the JD but not in the resume.
3. Suggested additions or modifications to the resume to better target this role.
"""
    )
    chain = prompt | llm
    response = chain.invoke({"resume": resume_text, "jd": job_description})
    return response.content if hasattr(response, "content") else response

def setup_chat_memory():
    return ConversationBufferMemory(memory_key="history", return_messages=True)

def chat_with_resume(user_query, resume_text, memory, api_key):
    """Chat using the entire resume text in context."""
    llm = get_llm(api_key)
    context = resume_text

    # Simplified prompt – only context and query are needed
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="""You are an AI assistant helping a user with their resume.
Use the following pieces of context from the user's resume to answer their question.
If you don't know the answer, just say that you don't know based on the resume. Don't make up information.

Context from Resume:
{context}

User Query: {query}
Answer:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    response = chain.invoke({"context": context, "query": user_query})
    return response["text"]

