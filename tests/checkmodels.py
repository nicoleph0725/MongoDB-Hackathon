from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client_ai.models.list():
    print(model.name)