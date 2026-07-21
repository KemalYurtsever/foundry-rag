from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from pathlib import Path

from .database import SQLiteStore


LOGGER = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embeddings are optional but cannot currently be used."""


class FoundryEmbeddingProvider:
    """Cached embeddings generated in-process by a Foundry Local model."""

    def __init__(
        self,
        model_alias: str = "qwen3-embedding-0.6b",
        download: bool = False,
        cache_dir: str | Path = ".rag_cache",
        batch_size: int = 64,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        from foundry_local_sdk import Configuration, FoundryLocalManager

        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(Configuration(app_name="foundry-rag"))

        manager = FoundryLocalManager.instance
        if manager is None:
            raise RuntimeError("Foundry Local manager did not initialize")

        model = manager.catalog.get_model(model_alias)
        if model is None:
            raise ValueError(f"Foundry Local embedding model not found: {model_alias}")
        if not model.is_cached:
            if not download:
                raise EmbeddingUnavailableError(
                    f"Embedding model '{model_alias}' is not downloaded; using lexical retrieval"
                )
            model.download()

        self._model = model
        self._loaded_here = not model.is_loaded
        if self._loaded_here:
            model.load()

        self.client = model.get_embedding_client()
        self.model_alias = model_alias
        self.model_key = self._model_cache_key(model_alias, model)
        self.store = SQLiteStore(Path(cache_dir) / "rag.sqlite3")
        self.dimension = self.store.get_model_dimension(self.model_key)
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self._closed = False

    @staticmethod
    def _model_cache_key(model_alias: str, model: object) -> str:
        """Include resolved model metadata so upgraded aliases do not reuse stale vectors."""
        info = getattr(model, "info", None)
        values = [model_alias]
        for owner, attribute in (
            (model, "id"),
            (model, "alias"),
            (model, "version"),
            (info, "id"),
            (info, "version"),
            (info, "variant"),
            (info, "provider"),
        ):
            value = getattr(owner, attribute, None) if owner is not None else None
            if value not in (None, ""):
                values.append(str(value))
        identity = "\0".join(values)
        return f"{model_alias}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"

    def _key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model_key}\0{text}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector or not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Embedding model returned an invalid vector")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._closed:
            raise RuntimeError("Embedding provider is closed")
        if not texts:
            return []

        keys = [self._key(text) for text in texts]
        cached = self.store.get_embeddings(self.model_key, list(dict.fromkeys(keys)))

        missing_by_key: dict[str, str] = {}
        for text, key in zip(texts, keys, strict=True):
            if key not in cached:
                missing_by_key.setdefault(key, text)

        missing_items = list(missing_by_key.items())
        completed_count = len(keys) - len(missing_items)
        progress_callback = getattr(self, "progress_callback", None)
        cancel_check = getattr(self, "cancel_check", None)
        if progress_callback:
            progress_callback(completed_count, len(keys))
        for start in range(0, len(missing_items), self.batch_size):
            if cancel_check and cancel_check():
                raise RuntimeError("Indexing cancelled")
            batch = missing_items[start : start + self.batch_size]
            response = self.client.generate_embeddings([text for _, text in batch])
            data = list(getattr(response, "data", []))
            if len(data) != len(batch):
                raise RuntimeError("Embedding model returned an unexpected number of vectors")

            batch_generated: dict[str, list[float]] = {}
            for (key, _), item in zip(batch, data, strict=True):
                vector = [float(value) for value in item.embedding]
                self._validate_vector(vector)
                if self.dimension is None:
                    self.dimension = len(vector)
                if len(vector) != self.dimension:
                    raise RuntimeError("Embedding model returned inconsistent vector dimensions")
                batch_generated[key] = vector
            if self.dimension is None:
                raise RuntimeError("Embedding model returned no vector dimension")
            # Persist each completed batch so cancellation or a later failure can
            # resume instead of regenerating all prior vectors.
            self.store.put_embeddings(self.model_key, self.dimension, batch_generated)
            cached.update(batch_generated)
            completed_count += len(batch)
            if progress_callback:
                progress_callback(min(completed_count, len(keys)), len(keys))

        vectors = [cached[key] for key in keys]
        if vectors and self.dimension is None:
            self.dimension = len(vectors[0])
        if self.dimension is None or any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("Embedding cache contains inconsistent vector dimensions")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._closed:
            raise RuntimeError("Embedding provider is closed")
        response = self.client.generate_embedding(text)
        data = list(getattr(response, "data", []))
        if len(data) != 1:
            raise RuntimeError("Embedding model returned an unexpected query response")
        vector = [float(value) for value in data[0].embedding]
        self._validate_vector(vector)
        if self.dimension is not None and len(vector) != self.dimension:
            raise RuntimeError("Query and document embedding dimensions do not match")
        if self.dimension is None:
            self.dimension = len(vector)
        return vector

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loaded_here:
            unload = getattr(self._model, "unload", None)
            if callable(unload):
                unload()

    def __enter__(self) -> "FoundryEmbeddingProvider":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def optional_foundry_embedder(
    model_alias: str,
    download: bool = False,
    cache_dir: str | Path = ".rag_cache",
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
):
    try:
        return FoundryEmbeddingProvider(
            model_alias,
            download,
            cache_dir,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    except (ImportError, EmbeddingUnavailableError) as exc:
        if download:
            raise
        LOGGER.warning("Embeddings disabled: %s", exc)
        return None
