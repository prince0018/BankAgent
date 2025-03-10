from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

load_dotenv()

# Update vector store path to use the current directory
persist_dir = "./vector_store"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Update to use the same model as other files
    google_api_key=GOOGLE_API_KEY
)

# Set up Gemini embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

def get_retriever():
    """Initialize and return the retriever"""
    if not Path(persist_dir).exists():
        raise ValueError("No vector store found. Please ingest documents first.")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embed_model
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

def search_documents(query: str) -> str:
    """
    Search through the documents using the vector store
    """
    try:
        # Load the vector store
        persist_dir = os.path.join(os.path.dirname(__file__), "vector_store")
        
        # Initialize embeddings
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Load the vector store
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embed_model
        )
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve more documents for better context
        )
        
        # Get relevant documents
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant information found."
            
        # Create a response using the LLM
        template = """You are a helpful banking assistant specializing in interest rates and account information.
        Use the following context to answer the question about interest rates.
        
        Important instructions:
        1. If the question asks about a future date, use the MOST RECENT interest rate information available
        2. Clearly state the interest rate tiers and conditions
        3. If the exact date isn't found, use the latest applicable rate and mention when it was last updated
        4. Always mention any important conditions or requirements
        
        Format your response:
        - Use £ symbol for currency values
        - Round interest rates to 2 decimal places
        - Present interest rates as percentages (e.g., 1.25%)
        - Use bullet points for multiple tiers
        - Include the effective date of the rates
        
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
                question=query
            )
        )
        
        return response.content
        
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Example usage:
# query = "What's the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?"
# response = search_documents(query)
# print(response) 