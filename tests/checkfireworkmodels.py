import os
import requests
from dotenv import load_dotenv

load_dotenv()

headers = {
    "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY')}",
    "Content-Type": "application/json"
}

response = requests.get(
    "https://api.fireworks.ai/inference/v1/models",
    headers=headers
)

models = response.json()
for model in models.get("data", []):
    print(model["id"])