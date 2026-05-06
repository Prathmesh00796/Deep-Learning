from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(documents):
    """
    Takes LangChain Document objects, splits them into chunks,
    generates embeddings, and creates a local FAISS vector store.
    """
    if not documents:
        return None

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings (Local HuggingFace model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store
