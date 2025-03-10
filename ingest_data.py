import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_parse import LlamaParse
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import json
from langchain_core.documents import Document

load_dotenv()

# Update file paths to point to parent directory
file_paths = [
    os.path.join(os.path.dirname(__file__), "..", "docs", "interest-rates_1.pdf"),
    os.path.join(os.path.dirname(__file__), "..", "docs", "interest-rates_2.pdf"),
    os.path.join(os.path.dirname(__file__), "..", "docs", "interest-rates_3.pdf")
]
persist_dir = os.path.join(os.path.dirname(__file__), "vector_store")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Set up Gemini embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

def load_and_split_documents(file_paths: List[str]):
    """Load PDFs and split them into chunks"""
    documents = []
    llama_parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        num_workers=2,
        verbose=True,
        language="en"
    )
    
    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")
            # Load and parse the document
            parsed_docs = llama_parser.load_data(file_path)
            if parsed_docs:
                # Convert llama_parse documents to langchain documents
                for doc in parsed_docs:
                    content = doc.get_content()
                    # Add specific metadata to help with retrieval
                    metadata = {
                        "source": file_path,
                        "type": "interest_rates",
                        "product": "Cash ISA Saver"
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
                print(f"Successfully processed {file_path}")
            else:
                print(f"Warning: No content extracted from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not documents:
        raise ValueError("No documents were successfully loaded")
    
    # Use smaller chunks with more overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,  # More overlap
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from {len(documents)} documents")
    return split_docs

def ingest_documents(file_paths: List[str], persist_dir: str):
    """
    Ingest documents and create a vector store
    """
    print("Loading and splitting documents...")
    documents = load_and_split_documents(file_paths)
    
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    # Test the ingestion with a sample query
    print("\nTesting vector store retrieval...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    test_query = "What are the interest rates for Cash ISA Saver accounts?"
    docs = retriever.invoke(test_query)
    print(f"Found {len(docs)} relevant documents for test query")
    
    return vectorstore

if __name__ == "__main__":
    # Ingest the PDF files into the vector store
    vectorstore = ingest_documents(file_paths, persist_dir)
    
    # Test the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    question = "What's the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?"
    docs = retriever.invoke(question)
    
    # Create a response using the LLM
    from langchain_core.prompts import ChatPromptTemplate
    
    template = """You are a helpful banking assistant. Use the following context to answer the question.
    If you don't know the answer, just say you don't know.

    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get response from LLM
    response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )
    
    print("\nTest Question:", question)
    print("Response:", response.content) 