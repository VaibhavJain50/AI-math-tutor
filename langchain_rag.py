import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def create_vector_store(pdf_path, db_dir="vectorstore/"):
    """
    Creates and saves a FAISS vector store from a PDF document.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} document pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    os.makedirs(db_dir, exist_ok=True)
    vectordb.save_local(db_dir)
    print(f"Vector store saved to {db_dir}")
    return vectordb


def load_qa_chain(db_dir="vectorstore/"):
    """
    Loads the FAISS vector store and creates a strict RetrievalQA chain.
    """
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectordb = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load vector store from {db_dir}. Did you run create_vector_store?")
        raise e

    # Initialize Groq LLM
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

    # Define strict prompt
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly and patient math tutor.

Use ONLY the information provided in the context below. 
Do NOT use outside knowledge, assumptions, or formulas that are not shown in the context.

-----------------------
CONTEXT:
{context}
-----------------------

QUESTION:
{question}

-----------------------

YOUR TASK:
1. Explain the solution **step-by-step**, slowly and clearly, as if teaching a beginner.
2. After explaining the reasoning, state the final answer at the end in this format:
   **Final Answer:** <answer>

If the context does NOT contain the information needed to answer, reply exactly with:
**The answer is not in the document.**
"""
)


    # Create QA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain