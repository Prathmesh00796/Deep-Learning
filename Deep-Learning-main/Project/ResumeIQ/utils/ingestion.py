import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def parse_uploaded_file(uploaded_file):
    """
    Saves an uploaded Streamlit file to a temporary file,
    loads it using the appropriate LangChain loader,
    and returns a list of Document objects.
    """
    if not uploaded_file:
        return []

    # Save to a temp file
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        documents = []
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return documents
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
