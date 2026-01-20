import os
from dotenv import load_dotenv
from google import genai

# Load gemini.env file
load_dotenv(".env")

API_KEY = os.getenv("GEMI_API_KEY")
print("API Key loaded:", API_KEY)
if not API_KEY:
    raise ValueError("GEMI_API_KEY not found. Check .env file")

client = genai.Client(api_key=API_KEY)
