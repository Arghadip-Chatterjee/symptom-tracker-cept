from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

PLACEHOLDER_KEYS = {"", "sk-your-key-here", "changeme", "your-api-key"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    admin_api_key: str = "change-me-admin-secret"
    chroma_path: str = str(REPO_ROOT / "data" / "chroma")
    upload_dir: str = str(REPO_ROOT / "data" / "uploads")
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'symptom_tracker.db'}"
    cors_origins: str = "http://localhost:3000"
    collection_name: str = "medical_knowledge"
    top_k: int = 6
    retrieve_k: int = 12
    vector_candidate_k: int = 24
    bm25_candidate_k: int = 24
    rrf_k: int = 60
    min_rerank_score: float = -2.0
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_l2_distance: float = 1.25
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma_path).resolve()

    @property
    def openai_configured(self) -> bool:
        key = (self.openai_api_key or "").strip()
        return bool(key) and key.lower() not in PLACEHOLDER_KEYS


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
