import os
from dotenv import load_dotenv
from google import genai

# Load gemini.env file
load_dotenv("gemini.env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check gemini.env file")

client = genai.Client(api_key=API_KEY)
