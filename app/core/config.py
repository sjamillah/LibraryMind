import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    AMALI_GATEWAY_URL: str = os.getenv("AMALI_GATEWAY_URL", "")
    AMALI_API_KEY: str = os.getenv("AMALI_API_KEY", "")
    PRIMARY_PROVIDER: str = os.getenv("PRIMARY_PROVIDER", "openai")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.4"))

    OPENAI_API_MODEL: str = "gpt-3.5-turbo"
    ANTHROPIC_API_MODEL: str = "claude-3-5-sonnet-20241022"


settings = Settings()

if not settings.AMALI_API_KEY:
    raise RuntimeError("AMALI_API_KEY must be set in .env")

if not settings.AMALI_GATEWAY_URL:
    raise RuntimeError("AMALI_GATEWAY_URL must be set in .env")
