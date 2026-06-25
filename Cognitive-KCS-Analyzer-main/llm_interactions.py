import google.genai as genai
from config import API_KEY, LLM_MODEL
# from tenacity import retry, stop_after_attempt, wait_random_exponential # For robust API calls

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_llm_completion(prompt_text, model=LLM_MODEL, max_tokens=1000, temperature=0.3):
    """
    Gets a completion from the specified Gemini LLM model.
    Uses google.genai to generate content.
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error: Could not get response from LLM. API Error: {str(e)}"

if __name__ == '__main__':
    # Test the function (requires your .env file and API key)
    # Ensure config.py correctly loads your API_KEY
    if API_KEY and API_KEY != "your_gemini_api_key_here":
        test_prompt = "Explain what a Large Language Model is in one sentence."
        print(f"Testing LLM with model: {LLM_MODEL}")
        explanation = get_llm_completion(test_prompt, model=LLM_MODEL, max_tokens=150)
        print("LLM Response:")
        print(explanation)
    else:
        print("Skipping LLM test as API_KEY is not configured or is placeholder.")
        print("Please set your GEMINI_API_KEY in a .env file.")
