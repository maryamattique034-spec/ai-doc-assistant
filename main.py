
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API Key from .env
load_dotenv()

# Initialize Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Control Temperature & Sampling Parameters
config = types.GenerateContentConfig(
    temperature=0.3, # Low temperature for more factual, precise answers
    top_p=0.8,
    system_instruction="You are a strict technical assistant. Keep answers concise and direct."
)

# Step 2: Generate Response
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents='Explain what a Vector Database is in two sentences.',
    config=config
)

print(response.text)