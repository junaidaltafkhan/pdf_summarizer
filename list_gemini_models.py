# list_gemini_models.py
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

def list_models():
    try:
        resp = client.models.list()
        print("---- Available Models ----\n")
        for m in resp:
            print(m.name if hasattr(m, "name") else m)
        print("\n--------------------------")
    except Exception as e:
        print("Error listing models:", e)

if __name__ == "__main__":
    list_models()
