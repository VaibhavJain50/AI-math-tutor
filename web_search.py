# web_search.py
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    print("Warning: TAVILY_API_KEY not set in environment. Web search will fail.")
    client = None
else:
    client = TavilyClient(api_key=API_KEY)

def web_search(query: str, max_tokens=1500):
    """
    Performs an optimized web search for a math problem.
    """
    if not client:
        print("Tavily client not initialized. Check your API key.")
        return None

    # Optimize the query for math solutions
    optimized_query = f"step-by-step solution for: {query}"

    try:
        response = client.search(
            query=optimized_query,
            include_raw_content=False, # We just want the processed content
            max_results=3 # Get top 3 relevant results
        )

        # Collect text chunks
        results = [item["content"] for item in response["results"] if "content" in item]
        
        return "\n\n---\n\n".join(results) if results else None
    
    except Exception as e:
        print(f"Tavily search error: {e}")
        return None