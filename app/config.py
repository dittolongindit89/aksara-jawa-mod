import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API Key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY in the .env file.")
