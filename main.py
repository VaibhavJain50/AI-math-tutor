import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# Your project files
from langchain_rag import load_qa_chain
from web_search import web_search

# Load environment variables
load_dotenv()

# --- Load the new RAG chain (from langchain_rag.py) ---
# This chain is "strict" and will be our Step 1.
try:
    strict_qa_chain = load_qa_chain()
except Exception as e:
    print(f"CRITICAL ERROR: Could not load QA chain. {e}")
    strict_qa_chain = None

# --- Load a "fallback" LLM ---
# This LLM will be used for Step 2 (Web) and Step 3 (Pure Reasoning).
# It uses a "tutor" prompt, not the strict RAG prompt.
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

def _build_fallback_prompt(query: str, reference_text: str) -> str:
    """
    Creates the "tutor" prompt for the fallback LLM.
    The model MUST give a step-by-step explanation.
    If information is missing from the reference, return "No solution found".
    """
    reference_block = f"REFERENCE (Use only this information):\n{reference_text}\n" if reference_text else ""

    prompt = f"""
You are a friendly math tutor. You MUST answer strictly based on the REFERENCE.
Do not use any outside knowledge. If the REFERENCE does not contain enough information,
respond with exactly: "No solution found"

{reference_block}

QUESTION:
{query}

### INSTRUCTIONS FOR YOUR RESPONSE:
1. First, restate the problem in simple words.
2. Then show **each step of the solution** in a slow and clear manner.
3. Explain why each step is performed.
4. At the end, state the final answer clearly in a separate line like:
Final Answer: <answer>

If the REFERENCE does not allow finding the answer:
Return exactly: "No solution found"
"""

    return prompt.strip()



def _looks_like_solution_text(text: str) -> bool:
    """
    This check is still useful for validating web search results
    and the final LLM output.
    """
    if not text:
        return False
    
    t = text.lower()
    solution_keywords = ["solution:", "step 1", "step 2", "answer:", "=>", "therefore"]
    has_keyword = any(ind in t for ind in solution_keywords)
    has_equals = "=" in t
    has_number = any(char.isdigit() for char in t)
    has_calculation = has_equals and has_number
    has_steps = t.count("step ") > 1 or t.count("\n- ") > 2

    return has_keyword or has_calculation or has_steps

# --- The New Solver Function ---
# --- The New Solver Function ---
def solve_math_question(query):
    kb_used = False
    web_used = False

    # --- 1. Try Knowledge Base (Strict) ---
    if strict_qa_chain:
        try:
            # The 'query' key is the default for RetrievalQA
            result = strict_qa_chain.invoke({"query": query})
            answer = result['result'].strip()

            # This is our new, reliable trigger!
            if "the answer is not in the document" not in answer.lower():
                kb_used = True
                return answer, kb_used, web_used
            
            # If we're here, the KB failed, so we fall through...
        except Exception as e:
            print(f"Error during KB lookup: {e}")
            # Fall through to web search

    # --- 2. Fallback to Web Search ---
    reference = None
    web_data = web_search(query)
    
    # --- FIX 1: Be less strict. If Tavily found *anything*, use it. ---
    if web_data:
        web_used = True
        reference = web_data
    
    # --- 3. Fallback to LLM (with or without web reference) ---
    try:
        # Build the "tutor" prompt, passing web data if we have it
        prompt = _build_fallback_prompt(query, reference)
        
        # Invoke the fallback LLM
        response = fallback_llm.invoke(prompt)
        answer = response.content.strip()

        # --- FIX 2: Simplify the final check. ---
        # Don't use _looks_like_solution_text here. It's too strict.
        # Trust the LLM unless the answer is obviously a refusal or empty.
        if not answer or "no solution found" in answer.lower() or len(answer) < 10:
            return "No solution found...", kb_used, web_used

        # If the answer is reasonable, return it.
        return answer, kb_used, web_used

    except Exception as e:
        print(f"Error during fallback LLM: {e}")
        return "No solution found....", kb_used, web_used

# --- Feedback logging (unchanged) ---
def store_feedback(question, answer, feedback):
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Question: {question}\nAnswer: {answer}\nFeedback: {feedback}\n---\n")

# --- Main (unchanged) ---
if __name__ == "__main__":
    if not strict_qa_chain:
        print("Exiting: QA chain could not be loaded.")
        exit()
        
    print("✅ Math Tutor Ready!\n")
    while True:
        q = input("🧮 Ask a math question: ")
        if q.lower() in ["quit", "exit"]:
            break
        ans, kb, web = solve_math_question(q)
        print("\n📘 Answer:\n", ans)
        
        if kb:
            print("[Source: Knowledge Base]")
        elif web:
            print("[Source: Web Search]")
        else:
            print("[Source: LLM Reasoning]")
        print("\n" + "---\n")

from langchain_google_genai import ChatGoogleGenerativeAI

classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

def is_math_question(query: str) -> bool:
    prompt = f"""
    You are a classifier. Decide if the following user query is a mathematics-related question.

    Return only "YES" or "NO".
    
    Query: {query}
    """
    result = classifier_llm.invoke(prompt).content.strip().upper()
    return result.startswith("Y")

