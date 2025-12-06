import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

CHROMA_PATH = "chroma_db"
TEXT_FILE_PATH = "Tindimenu_Customer_View_Summary.txt"

def load_text_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_knowledge_base():
    print(f"📄 Loading: {TEXT_FILE_PATH}")
    text = load_text_file(TEXT_FILE_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    print(f"💾 Knowledge base saved at: {CHROMA_PATH}")

if __name__ == "__main__":
    create_knowledge_base()
