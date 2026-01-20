import os
from flask.cli import load_dotenv
from google.genai import Client


# Load gemini.env file
load_dotenv(".env")
# Make sure your new key is set in environment variables
API_KEY = os.getenv("GEMI_API_KEY")
print("Using API Key:", API_KEY)
client = Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain Gemini 2.5 Flash in one line."
)

print(response)
