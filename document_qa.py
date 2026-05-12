import os
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-2.5-flash"

MAX_DOC_CHARS = 100000
MAX_HISTORY_TURNS = 6

# ----------------------------
# LOAD DOCUMENT
# ----------------------------
def load_full_document(path):
    print("Loading PDF...")

    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text()

    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]

    print(f"Loaded document ({len(text)} chars)")
    return text


PDF_FILES = {
    "manual_a": "sources/SuperVario_Deutsch_LAS4.pdf",
    "manual_b": "sources/POLARO_M_L_XL_march2024.pdf"
}

documents = {
    key: load_full_document(path)
    for key, path in PDF_FILES.items()
}
# ----------------------------
# SESSION MEMORY
# ----------------------------
sessions = {}  # {session_id: [{"user": ..., "bot": ...}, ...]}


def get_session_history(session_id):
    if isinstance(session_id, list):
        session_id = session_id[0]

    if session_id not in sessions:
        sessions[session_id] = []

    return sessions[session_id]


def add_to_history(session_id, user_msg, bot_msg):
    history = get_session_history(session_id)

    history.append({
        "user": user_msg,
        "bot": bot_msg
    })

    if len(history) > MAX_HISTORY_TURNS:
        history.pop(0)


def format_history(session_id):
    history = get_session_history(session_id)

    text = ""
    for h in history:
        text += f"User: {h['user']}\n"
        text += f"Assistant: {h['bot']}\n\n"

    return text



# ----------------------------
# STREAM QA FUNCTION
# ----------------------------
def stream_gemini(prompt):
    try:
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt
        )

        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    except Exception as e:
        print("Streaming error:", e)
        yield "\n[Error: model unavailable]"
        
        
def ask_gemini_stream(query, session_id, manual):        
    history_text = format_history(session_id)
    user_manual = documents.get(manual)
    prompt = f"""
            You are an expert assistant answering questions based ONLY on the user manual.
            
            Rules:
            - If the answer is not in the document, say something helpful.
            - Do NOT invent values
            - Be precise with numbers (especially tables)
            - Answer in the same language as the QUESTION (which might be different from the user manual language)
            
            CHAT HISTORY:
            {history_text}
            
            USER MANUAL:
            {user_manual}
            
            QUESTION:
            {query}
            """

    full_answer = ""

    for token in stream_gemini(prompt):
        full_answer += token
        yield token

    # store after full answer is done
    if full_answer and not full_answer.startswith("Error"):
        add_to_history(session_id, query, full_answer)        
# ----------------------------
# BACKWARD COMPATIBILITY
# ----------------------------
def retrieve_chunks(query, top_k=5):
    return [full_text]