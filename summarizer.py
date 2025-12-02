# summarizer.py
import os
import hashlib
from dotenv import load_dotenv
import fitz  # PyMuPDF
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

MODEL_ID = "models/gemini-2.5-flash"


def file_hash_bytes(bts: bytes) -> str:
    return hashlib.sha256(bts).hexdigest()


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages).strip()


def chunk_text(text: str, max_chars: int = 3000):
    text = text.replace("\r\n", "\n").strip()
    n = len(text)

    if n <= max_chars:
        return [text]

    chunks = []
    i = 0
    while i < n:
        end = min(i + max_chars, n)
        candidate = text[i:end]

        split_pos = max(candidate.rfind("\n"), candidate.rfind("."))

        if split_pos > max_chars // 2:
            end = i + split_pos + 1

        chunks.append(text[i:end].strip())
        i = end

    return chunks


def summarize_chunk(chunk: str, target_words: int = 200) -> str:
    prompt = (
        f"Summarize the following text in about {target_words} words, clearly and concisely.\n\n"
        f"{chunk}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini summarize_chunk error: {e}")


def aggregate_summaries(summaries, target_words: int = 300) -> str:
    combined = "\n\n".join(summaries)

    prompt = (
        f"Combine these summaries into one final summary of about {target_words} words.\n\n"
        f"{combined}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini aggregate_summaries error: {e}")
