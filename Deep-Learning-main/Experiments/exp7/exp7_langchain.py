import os
import getpass

def run_experiment():
    print("--- Experiment 7: Langchain with Groq API ---")
    
    # Prompt for API Key at runtime (no initialization in code)
    if "GROQ_API_KEY" not in os.environ:
        api_key = getpass.getpass("Enter your Groq API Key: ")
        os.environ["GROQ_API_KEY"] = api_key
        
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        print("Required libraries are not installed. Please install them using:")
        print("pip install langchain-groq langchain-core")
        return

    # Initialize the Groq Chat Model
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=512
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant specialized in explaining Deep Learning concepts."),
        ("human", "{question}")
    ])

    # Create a simple chain
    chain = prompt | llm

    # Ask a question
    question = "Can you briefly explain the difference between RNN and LSTM in one paragraph?"
    print(f"\nQuestion: {question}")
    print("\nQuerying Groq API...")
    
    try:
        response = chain.invoke({"question": question})
        print("\nResponse:")
        print(response.content)
    except Exception as e:
        print(f"\nError occurred while calling the API: {e}")

    print("\nExperiment 7 Complete.")

if __name__ == "__main__":
    run_experiment()
