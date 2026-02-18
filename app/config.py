import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env")

@dataclass(frozen=True)
class Settings:
    telegram_token: str = os.environ["TELEGRAM_TOKEN"]
    hf_api_key: str = os.environ["HF_API_KEY"]
    llm_model_name: str = os.environ["LLM_MODEL_NAME"]

    pg_host: str = os.environ["PGHOST"]
    pg_port: int = int(os.environ.get("PGPORT", "5432"))
    pg_db: str = os.environ["PGDB"]
    pg_user: str = os.environ["PGUSER"]
    pg_password: str = os.environ["PGPASSWORD"]

    embed_model_name: str = os.environ.get(
        "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.environ.get("TOP_K", "5"))

settings = Settings()
