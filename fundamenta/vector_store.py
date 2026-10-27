"""Módulo 05: almacén vectorial.

Guarda los embeddings de los chunks en una matriz de NumPy y busca los más parecidos a
una consulta. Como todos los vectores están normalizados, la similitud coseno de la
consulta con todos los chunks se calcula con un único producto matriz-vector.

Esta búsqueda exacta ("fuerza bruta") es suficiente para decenas de miles de chunks.
Para millones, las bases de datos vectoriales usan índices aproximados (ver módulo 11).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fundamenta.embeddings import normalize_rows
from fundamenta.loaders import Document


def matches(document: Document, where: dict[str, Any] | None) -> bool:
    """Indica si los metadatos del documento cumplen todas las condiciones de `where`."""
    return not where or all(document.metadata.get(key) == value for key, value in where.items())


class VectorStore:
    """Almacén de embeddings en memoria con búsqueda exacta y filtros por metadatos."""

    def __init__(self) -> None:
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.documents: list[Document] = []

    def __len__(self) -> int:
        return len(self.documents)

    def add(self, vectors: np.ndarray, documents: list[Document]) -> None:
        """Añade vectores (uno por documento). Se normalizan al guardarlos."""
        vectors = normalize_rows(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or len(vectors) != len(documents):
            raise ValueError("Se necesita exactamente un vector por documento")
        if len(self) and vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Dimensión incorrecta: {vectors.shape[1]} (se esperaba {self.vectors.shape[1]})"
            )
        self.vectors = vectors if not len(self) else np.vstack([self.vectors, vectors])
        self.documents.extend(documents)

    def search(
        self, query: np.ndarray, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[int, float]]:
        """Devuelve los `k` documentos más similares como `(índice, similitud)`.

        `where` filtra por metadatos, por ejemplo `{"source": "manual-del-empleado.md"}`.
        """
        if not len(self):
            return []
        query = np.asarray(query, dtype=np.float32)
        scores = self.vectors @ (query / (np.linalg.norm(query) or 1))
        if where:
            allowed = np.array([matches(doc, where) for doc in self.documents])
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def save(self, directory: str | Path) -> None:
        """Guarda los vectores (`vectors.npy`) y los documentos (`documents.json`)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = [asdict(doc) for doc in self.documents]
        (directory / "documents.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """Carga un almacén guardado con `save`."""
        directory = Path(directory)
        store = cls()
        store.vectors = np.load(directory / "vectors.npy")
        payload = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        store.documents = [Document(**item) for item in payload]
        return store
