"""Módulo 06: recuperación híbrida.

Combina la búsqueda léxica (BM25), que es precisa con nombres propios, cifras y
términos exactos, con la búsqueda semántica (embeddings), que entiende sinónimos y
paráfrasis. Los dos rankings se fusionan con Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fundamenta.bm25 import BM25Index
from fundamenta.chunking import contextualize
from fundamenta.embeddings import Embedder
from fundamenta.loaders import Document
from fundamenta.vector_store import VectorStore, matches

Mode = Literal["lexical", "semantic", "hybrid"]


@dataclass
class SearchResult:
    """Un chunk recuperado, con su puntuación y su posición en el ranking (desde 1)."""

    document: Document
    score: float
    rank: int


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Fusiona varios rankings sumando `peso / (k + posición)` para cada documento.

    Solo usa las posiciones, no las puntuaciones, por lo que no hace falta que las
    puntuaciones de BM25 y del coseno estén en la misma escala. `k = 60` es el valor
    propuesto en el artículo original (Cormack et al., 2009). Los pesos permiten dar
    más influencia a uno de los rankings (por defecto, todos pesan 1).
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + weight / (k + position)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def mmr(
    query: np.ndarray,
    vectors: np.ndarray,
    candidates: list[int],
    k: int,
    diversity: float = 0.3,
) -> list[int]:
    """Maximal Marginal Relevance: reordena candidatos evitando resultados redundantes.

    En cada paso elige el candidato que maximiza
    `(1 - diversity) * similitud_con_consulta - diversity * similitud_con_ya_elegidos`.
    """
    selected: list[int] = []
    remaining = list(candidates)
    relevance = {i: float(vectors[i] @ query) for i in remaining}
    while remaining and len(selected) < k:

        def gain(i: int) -> float:
            redundancy = max((float(vectors[i] @ vectors[j]) for j in selected), default=0.0)
            return (1 - diversity) * relevance[i] - diversity * redundancy

        best = max(remaining, key=gain)
        selected.append(best)
        remaining.remove(best)
    return selected


class Retriever:
    """Busca los chunks más relevantes para una consulta.

    - `mode="lexical"`: solo BM25. No necesita embedder.
    - `mode="semantic"`: solo embeddings.
    - `mode="hybrid"`: ambos, fusionados con RRF (recomendado).

    `contextual=True` indexa cada chunk junto con su sección (ver `contextualize`).
    `lexical_weight` es el peso de BM25 en la fusión híbrida (el semántico pesa 1). El
    valor por defecto, 0.5, es el que mejor resultado dio en la evaluación del módulo 10.
    """

    def __init__(
        self,
        chunks: list[Document],
        embedder: Embedder | None = None,
        mode: Mode = "hybrid",
        contextual: bool = True,
        candidates: int = 20,
        rrf_k: int = 60,
        lexical_weight: float = 0.5,
    ) -> None:
        if mode != "lexical" and embedder is None:
            raise ValueError(f"El modo {mode!r} necesita un embedder")
        self.chunks = chunks
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight

        to_text: Callable[[Document], str] = contextualize if contextual else lambda c: c.text
        texts = [to_text(chunk) for chunk in chunks]
        self.bm25 = BM25Index().fit(texts)
        self.store = VectorStore()
        if embedder is not None:
            fit = getattr(embedder, "fit", None)
            if callable(fit):
                fit(texts)
            self.store.add(embedder.embed_documents(texts), chunks)

    def _lexical(self, query: str, where: dict[str, Any] | None) -> list[tuple[int, float]]:
        ranked = self.bm25.search(query, k=len(self.chunks))
        return [(i, s) for i, s in ranked if matches(self.chunks[i], where)][: self.candidates]

    def _semantic(
        self, query_vector: np.ndarray, where: dict[str, Any] | None
    ) -> list[tuple[int, float]]:
        return self.store.search(query_vector, k=self.candidates, where=where)

    def search(
        self,
        query: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        """Devuelve los `k` chunks más relevantes.

        `where` filtra por metadatos. `diversity` (entre 0 y 1) activa MMR para evitar
        devolver varios chunks casi idénticos; requiere un embedder.
        """
        query_vector = self.embedder.embed_query(query) if self.embedder else None
        if self.mode == "lexical":
            ranked = self._lexical(query, where)
        elif self.mode == "semantic":
            ranked = self._semantic(query_vector, where)
        else:
            lexical = [i for i, _ in self._lexical(query, where)]
            semantic = [i for i, _ in self._semantic(query_vector, where)]
            ranked = reciprocal_rank_fusion(
                [lexical, semantic], k=self.rrf_k, weights=[self.lexical_weight, 1.0]
            )

        indices = [i for i, _ in ranked]
        if diversity is not None and query_vector is not None:
            indices = mmr(query_vector, self.store.vectors, indices, k, diversity)
        scores = dict(ranked)
        return [
            SearchResult(document=self.chunks[i], score=scores[i], rank=position)
            for position, i in enumerate(indices[:k], start=1)
        ]
