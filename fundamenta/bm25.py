"""Módulo 03: búsqueda léxica con BM25.

BM25 puntúa cada documento según cuántas veces aparecen en él los términos de la
consulta, dando más peso a los términos poco frecuentes en la colección (IDF) y
corrigiendo por la longitud del documento. Es el algoritmo de ranking clásico de los
buscadores y sigue siendo una base muy sólida en sistemas RAG.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

from fundamenta.text import tokenize


class BM25Index:
    """Índice BM25 sobre una lista de textos.

    - `k1` controla la saturación de la frecuencia: a partir de cierto número de
      apariciones, repetir un término apenas suma.
    - `b` controla cuánto se penalizan los documentos largos (0 = nada, 1 = totalmente).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.term_frequencies: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.average_length = 0.0

    def fit(self, texts: list[str]) -> BM25Index:
        """Construye el índice: frecuencias por documento, longitudes e IDF."""
        self.term_frequencies = [Counter(self.tokenizer(text)) for text in texts]
        self.lengths = [sum(tf.values()) for tf in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(texts) if texts else 0.0

        document_frequency: Counter[str] = Counter()
        for tf in self.term_frequencies:
            document_frequency.update(tf.keys())
        n = len(texts)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        return self

    def __len__(self) -> int:
        return len(self.term_frequencies)

    def scores(self, query: str) -> list[float]:
        """Devuelve la puntuación BM25 de la consulta para cada documento del índice."""
        terms = self.tokenizer(query)
        results = []
        for tf, length in zip(self.term_frequencies, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
            results.append(score)
        return results

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Devuelve los `k` documentos con mayor puntuación como `(índice, puntuación)`.

        Los documentos que no comparten ningún término con la consulta se excluyen.
        """
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:k] if score > 0]
