import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings as ChromaClientSettings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings


def _silence_chroma_telemetry() -> None:
    """Chroma 0.5.x posthog telemetry is broken (capture arity). Disable it."""
    try:
        from chromadb.telemetry.product import posthog as chroma_posthog

        def _noop_capture(self, *args, **kwargs):  # noqa: ANN001
            return None

        chroma_posthog.Posthog.capture = _noop_capture  # type: ignore[method-assign]
    except Exception:
        pass


_silence_chroma_telemetry()


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaClientSettings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
    )
