import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load gemini.env file
load_dotenv(".env")

API_KEY = os.getenv("GEMI_API_KEY") or os.getenv("GEMINI_API_KEY")
print("API Key loaded:", "***" if API_KEY else None)
if not API_KEY:
    raise ValueError("GEMI_API_KEY or GEMINI_API_KEY not found. Check gemini.env or .env file")

genai.configure(api_key=API_KEY)
client = genai.GenerativeModel('gemini-3-flash-preview')
