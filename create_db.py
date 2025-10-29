import sys
from langchain_rag import create_vector_store

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file_path = sys.argv[1]
        print(f"Creating vector store from: {pdf_file_path}")
        create_vector_store(pdf_file_path)
    else:
        print("Error: Please provide the path to your PDF file.")
        print("Usage: python create_db.py path/to/your/math-book.pdf")