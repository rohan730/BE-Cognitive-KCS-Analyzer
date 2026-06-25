GEMINI_API_KEY="your_gemini_api_key_here"

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")

# You can add other configurations here if needed
LLM_MODEL = "gemini-2.5-flash" # Gemini model for text generation
# Gemini API uses a simple generate_content method for completions
