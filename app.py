# -------------------------------------------------------------
# TINDIMENU FASTAPI — FINAL CLEANED VERSION
# With: LLM rotation + conversation history + controlled behavior
# -------------------------------------------------------------

import os
import re
import random
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uvicorn

from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
CURRENT_KEY_INDEX = 0

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

WHATSAPP_API = os.getenv("WHATSAPP_API", "").strip()
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "").strip()

CHROMA_PATH = "chroma_db"
DB_FILE = "user_memory.db"

app = FastAPI(title="Tindimenu")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# -------------------------------------------------------------
# DATABASE SETUP
# -------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            user_id TEXT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            warned_once INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS failed_keys (
            key_value TEXT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

init_db()

def save_message(uid, sender, msg):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (uid, sender, msg))
    conn.commit()
    conn.close()

def get_last_messages(uid, limit=6):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT sender, message
        FROM chat_history
        WHERE user_id=?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (uid, limit)).fetchall()
    conn.close()
    return rows[::-1]

def get_user_memory(uid):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT name, warned_once FROM user_memory WHERE user_id=?", (uid,)).fetchone()
    conn.close()
    if not row:
        return {"name": None, "warned": False}
    return {"name": row["name"], "warned": bool(row["warned_once"])}

def update_user_memory(uid, name=None, warned=None):
    current = get_user_memory(uid)
    if name is None:
        name = current["name"]
    if warned is None:
        warned = current["warned"]

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        INSERT INTO user_memory (user_id, name, warned_once)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET name=excluded.name, warned_once=excluded.warned_once
        """,
        (uid, name, int(warned)),
    )
    conn.commit()
    conn.close()

def log_failed_key(key, err):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO failed_keys (key_value, error_message) VALUES (?, ?)",
        (key, err[:200])
    )
    conn.commit()
    conn.close()
    masked = key[:10] + "..."
    send_whatsapp_alert(f"🚨 Gemini Key Failed\nKEY: {masked}\nERROR: {err}")

# -------------------------------------------------------------
# WHATSAPP
# -------------------------------------------------------------

def send_whatsapp_alert(msg: str):
    if not WHATSAPP_API or not WHATSAPP_NUMBER:
        return
    try:
        requests.post(WHATSAPP_API, json={"number": WHATSAPP_NUMBER, "message": msg}, timeout=10)
    except:
        pass

# -------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------

def get_gemini_llm():
    global CURRENT_KEY_INDEX
    for attempt in range(len(GEMINI_KEYS)):
        idx = (CURRENT_KEY_INDEX + attempt) % len(GEMINI_KEYS)
        key = GEMINI_KEYS[idx]
        os.environ["GOOGLE_API_KEY"] = key
        try:
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
            model.invoke("hello")  # validate
            CURRENT_KEY_INDEX = idx
            return model
        except Exception as e:
            log_failed_key(key, str(e))
    return None

llm = get_gemini_llm()
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
vs = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 5})

try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
except:
    emotion_classifier = None

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------

def detect_emotion(text):
    if not emotion_classifier:
        return "neutral"
    try:
        res = emotion_classifier(text)[0]
        return (res[0]["label"] if isinstance(res, list) else res["label"]).lower()
    except:
        return "neutral"


def build_system_prompt():
    return (
        "You are Tindi, a restaurant assistant chatbot and a friendly restaurant assistant, not a general AI.\n"
        "You ONLY do the following actions:\n"
        "1) Answer questions about the restaurant and its features.\n"
        "2) Respond politely, with a maximum of 1 emoji.\n"
        "3) If user asks for menu items, describe ONLY menu FEATURES, unless context provides actual items.\n"
        "4) If user asks for reservations, say you cannot book and tell them to call the restaurant.\n"
        "5) If user asks something outside food/dining, politely refuse.\n"
        "6) If user repeats a message, acknowledge it and avoid repeating the same thing.\n"
        "7) If context contains no real dishes, NEVER invent menu items.\n"
        "8) Keep answers short, clean, and realistic.\n"
        "9) If user is rude, stay calm and neutral.\n"
        "10) You ONLY remember name if code provides it — NEVER claim internal memory.\n"
        "11) You can respond casually and warmly in Telugu, Hindi, or English depending on the user's language.\n"
        "12) If user asks general friendly questions, reply warmly.\n"
        "13) Avoid adult or harmful content.\n"
        
    )

# -------------------------------------------------------------
# MAIN CHAT REASONING
# -------------------------------------------------------------

def generate_answer_raw(user_prompt, user_id):
    global llm

    if llm is None:
        return "I'm currently offline. Please try again later."

    memory = get_user_memory(user_id)
    lower = user_prompt.lower().strip()

   # --- Greetings (should NOT be treated as unclear) ---
    # --- Language detection ---
    def is_telugu(text):
        return any("\u0C00" <= ch <= "\u0C7F" for ch in text)

    def is_hindi(text):
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    language_instruction = ""

    if is_telugu(user_prompt):
        language_instruction = "Respond in Telugu."
    elif is_hindi(user_prompt):
        language_instruction = "Respond in Hindi."
    else:
        language_instruction = "Respond in English."

    greetings = {"hi", "hii", "hiii", "hello", "hey", "heyy", "heylo", "hola", "hlo"}
    if lower in greetings:
        greeting_responses = [
        "🙂 Hi there! How can I help you today?",
        "😊 Hey! What can I do for you?",
        "👋 Hello! How’s it going? What would you like to know?",
        "🙂 Hi! Ready to help — what do you need?"
        ]
        return random.choice(greeting_responses)


    # --- OK / acknowledgement handling ---
    ok_tokens = {"ok", "k", "kk", "okay", "okk", "oky"}
    if lower in ok_tokens:
        ok_responses = [
        "👌 Sure — want me to continue or explain something in detail?",
        "🙂 Alright! Just tell me what you need next.",
        "👌 Got it — should I go ahead?",
        "🙂 Okay! What would you like to do now?"
        ]
        return random.choice(ok_responses)
    # --- Thinking/filler sounds ---
    hmm_tokens = {"hmm", "mm", "mmm", "hmmm", "uh", "uhh", "huh"}
    if lower in hmm_tokens:
        hmm_responses = [
        "🙂 I’m here — take your time. Should I continue?",
        "😊 No rush — what would you like to do next?",
        "🙂 I'm listening — want me to guide you?",
        "👌 Sure — tell me whenever you're ready."
        ]
        return random.choice(hmm_responses)

    # --- Profanity handling ---
    bad_words = ["fuck", "shit", "bitch", "asshole"]
    if any(bad in lower for bad in bad_words):
        soft_responses = [
        "😔 I’m really sorry if something frustrated you. I’m here to help — tell me what went wrong.",
        "😔 It sounds like you're upset. I’m here to support you — what happened?",
        "😔 I get that something may have gone wrong. Let me help — what’s bothering you?",
        "😔 I’m sorry you're feeling like that. Tell me what’s wrong so I can help."
        ]
        return random.choice(soft_responses)

    # name storing
    if "my name is" in lower or "i am" in lower or "this is" in lower:
        name = user_prompt.split()[-1].strip("?!., ").capitalize()
        update_user_memory(user_id, name=name)
        return f"🙂 Hi {name}! Nice to meet you."

    if "my name" in lower:
        if memory["name"]:
            return f"🙂 You told me your name is {memory['name']}!"
        return "🙂 You haven't told me your name yet."

    # get conversation history
    history_rows = get_last_messages(user_id)
    conversation_history = ""
    for sender, msg in history_rows:
        role = "User" if sender == "user" else "Assistant"
        conversation_history += f"{role}: {msg}\n"

    # context
    context = ""
    try:
        docs = retriever.get_relevant_documents(user_prompt)
        context = "\n".join([d.page_content for d in docs[:3]])
    except:
        pass

    # described system behavior
    system_prompt = build_system_prompt()

    final_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation so far:\n{conversation_history}\n"
        f"Current message: {user_prompt}\n\n"
        f"Restaurant info:\n{context}"
    )

    try:
        response = llm.invoke(final_prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
    except Exception as e:
        log_failed_key(GEMINI_KEYS[CURRENT_KEY_INDEX], str(e))
        llm = get_gemini_llm()
        return "Sorry, I'm having trouble right now."

# -------------------------------------------------------------
# API
# -------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(default="guest")
    message: str

@app.post("/chat")
def chat(payload: ChatRequest):
    msg = payload.message.strip()
    if not msg:
        raise HTTPException(400, "Message empty")

    answer = generate_answer_raw(msg, payload.user_id)

    save_message(payload.user_id, "user", msg)
    save_message(payload.user_id, "bot", answer)

    return JSONResponse({"response": answer})

@app.get("/health")
def health():
    return {"status": "running", "model": "gemini" if llm else "fallback"}

@app.get("/", response_class=HTMLResponse)
def serve_html():
    file_path = BASE_DIR / "index.html"
    if not file_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return file_path.read_text(encoding="utf-8")

# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------

# --- 9️⃣ Server Start ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server at http://127.0.0.1:8001")
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)

