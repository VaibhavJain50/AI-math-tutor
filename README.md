🧮 AI Math Tutor Agent

This is an intelligent, multi-step AI Math Tutor built in Python using Streamlit, LangChain, and Groq.

The agent answers math-related questions using a 3-step fallback logic to provide the most accurate answer possible. It dynamically reports which source was used to generate the answer.

🌟 Features

Interactive Chat UI: A simple and clean user interface built with Streamlit.

3-Step Fallback Logic: The agent finds answers in a specific, prioritized order:

Local Knowledge Base: First, it performs a strict search against a local vector store (built from math_data.json) to find an exact, pre-verified answer.

Web Search: If no answer is found in the KB, it performs an optimized web search using Tavily AI to find relevant, real-time context.

LLM Reasoning: Finally, it uses a high-speed LLM (Groq Llama 3.1) to synthesize a final answer. This step will use the context from the web search if it was available, or solve the problem from its own knowledge if not.

Dynamic Source Reporting: The UI clearly labels the source of each answer:

Knowledge Used (Highest trust)

Tavily Web Search Used (Medium trust)

Pure Model Reasoning (No External Data) (Base trust)

Feedback Collection: Users can rate the helpfulness of each answer, and the feedback is logged to feedback_log.txt for review.

Customizable KB: Easily add new, verified math facts to your agent's "brain" by editing the math_data.json file and re-running the build script.

🛠️ Tech Stack & Architecture

Frontend: Streamlit

Core Logic: Python

LLM: Gemini

RAG (Knowledge Base):

LangChain (for RetrievalQA, JSONLoader)

FAISS (for the vector store)

Hugging Face Embeddings (all-MiniLM-L6-v2)

Web Search: Tavily AI

Dependencies: python-dotenv

🚀 Setup & Installation

1. Clone the Repository

git clone <your-repo-url>
cd <your-repo-name>


2. Install Dependencies

You will need to install all the required Python libraries.

pip install streamlit langchain langchain-community langchain-groq faiss-cpu sentence-transformers tavily-api python-dotenv jq


(Note: Use faiss-gpu if you have a compatible NVIDIA GPU)

3. Set Up Environment Variables

Create a file named .env in the root of your project directory and add your API keys:

GOOGLE_API_KEY=""
TAVILY_API_KEY="your-tavily-api-key"


4. Create Your Knowledge Base

This is a one-time setup step to build your local vectorstore/.

Edit your KB (Optional)


Run the Build Script: In your terminal, run the create_db.py script and pass it your pdf file.

python create_db.py math_data.pdf


You should see output indicating that the documents were loaded and the vector store was saved. You will now have a new vectorstore/ directory.

🏃‍♂️ How to Run

After completing the setup, run the Streamlit app:

streamlit run app.py


A browser window will automatically open to http://localhost:8501, and you can start asking math questions.

📄 File Structure

.
├── app.py                # The main Streamlit frontend application
├── main.py               # Contains the core `solve_math_question` logic and 3-step fallback
├── langchain_rag.py      # Handles loading the JSON and building/loading the RAG chain
├── create_db.py          # One-time script to build the FAISS vector store
├── web_search.py         # Manages the connection and query logic for Tavily AI
├── math_data.json        # Your customizable knowledge base
├── .env                  # Your secret API keys
└── feedback_log.txt      # (Auto-generated) Logs all user feedback
