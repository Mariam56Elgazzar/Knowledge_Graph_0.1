from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # ===== LLM =====
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # ===== Logging =====
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ===== Neo4j =====
    NEO4J_URL = os.getenv("NEO4J_URL", "")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


settings = Settings()

if not settings.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env")


