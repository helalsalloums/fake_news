from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from app.schemas.domain import Passage


class EmbeddingEncoder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str], *, query: bool = False) -> Any: ...


class SentenceTransformerEncoder:
    """Lazy multilingual encoder; it does not generate text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str], *, query: bool = False) -> Any:
        instruction = (
            "Given an Arabic news claim, retrieve passages that support or contradict it: "
            if query
            else ""
        )
        values = [f"{instruction}{text}" for text in texts]
        return self._load().encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class FaissVectorStore:
    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self.encoder = encoder
        self.passages: list[Passage] = []
        self.index: Any = None

    async def add(self, passages: Sequence[Passage]) -> None:
        import faiss

        vectors = self.encoder.encode([item.text for item in passages])
        self.index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.index.add(vectors.astype("float32"))
        self.passages = list(passages)

    async def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        vector = self.encoder.encode([query], query=True).astype("float32")
        scores, indexes = self.index.search(vector, min(limit, len(self.passages)))
        return [
            (self.passages[int(index)].id, max(0.0, min(1.0, float(score))))
            for index, score in zip(indexes[0], scores[0], strict=True)
            if index >= 0
        ]
