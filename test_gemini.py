import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key from .env
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Load Gemini 2.5 Flash model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Test prompt
response = model.generate_content(
    "Explain Gemini 2.5 Flash in one line."
)

print(response.text)