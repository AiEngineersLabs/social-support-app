from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://izharkhan@localhost:5432/social_support_db"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    LIGHT_LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # App
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    POLICY_DIR: str = str(BASE_DIR / "data" / "policies")

    model_config = {"env_file": str(BASE_DIR / ".env"), "extra": "ignore"}


settings = Settings()
