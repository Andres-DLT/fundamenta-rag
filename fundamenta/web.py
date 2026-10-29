"""Versión web sin servidor propio.

Fundamenta puede publicarse como una página estática más una función sin servidor:

- La búsqueda se ejecuta en el navegador de cada visitante (`web/engine.js`). Este
  módulo exporta todo lo que necesita en un único índice JSON: los fragmentos, las
  estadísticas de BM25 y los vectores calculados con el modelo E5 cuantizado.
- La generación con un LLM se ejecuta en una función sin servidor (`api/answer.py`). El
  navegador solo le envía la pregunta y los números de los fragmentos recuperados; la
  función reconstruye los fragmentos a partir de los mismos documentos y responde con
  el mismo `AnswerGenerator` que el servidor. Así la clave del proveedor nunca llega al
  navegador y la función no sirve para enviar textos arbitrarios al modelo.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from fundamenta.bm25 import BM25Index
from fundamenta.chunking import chunk_documents, contextualize
from fundamenta.embeddings import Embedder
from fundamenta.generation import Answer, AnswerGenerator
from fundamenta.llm import LLM, LLMError, get_llm
from fundamenta.loaders import Document, load_directory
from fundamenta.pipeline import PipelineConfig
from fundamenta.retriever import SearchResult
from fundamenta.server import MAX_QUESTION_LENGTH, RateLimiter, answer_to_dict
from fundamenta.text import SPANISH_STOPWORDS

# Modelo que el navegador usa para codificar las preguntas (transformers.js). Debe ser
# el mismo con el que se calculan aquí los vectores de los fragmentos (OnnxEmbedder).
BROWSER_MODEL = {"repo": "Xenova/multilingual-e5-small", "dtype": "int8", "prefix": "query: "}


def load_chunks(data_dir: str | Path, config: PipelineConfig | None = None) -> list[Document]:
    """Fragmentos de una colección, segmentados igual que en `RAGPipeline`."""
    config = config or PipelineConfig()
    return chunk_documents(load_directory(data_dir), config.chunk_size, config.overlap)


def corpus_fingerprint(chunks: list[Document]) -> str:
    """Huella de los fragmentos: el navegador y la función deben usar la misma colección."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(contextualize(chunk).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def build_index(
    chunks: list[Document],
    embedder: Embedder,
    config: PipelineConfig | None = None,
    examples: list[str] | None = None,
) -> dict:
    """Índice completo para la búsqueda en el navegador."""
    config = config or PipelineConfig()
    texts = [contextualize(chunk) if config.contextual else chunk.text for chunk in chunks]
    bm25 = BM25Index().fit(texts)
    fit = getattr(embedder, "fit", None)  # los embedders entrenables (LSA), como en Retriever
    if callable(fit):
        fit(texts)
    vectors = embedder.embed_documents(texts)
    return {
        "organization": config.organization,
        "examples": examples or [],
        "fingerprint": corpus_fingerprint(chunks),
        "model": BROWSER_MODEL,
        "retrieval": {
            "k": config.k,
            "candidates": 20,
            "rrf_k": 60,
            "lexical_weight": config.lexical_weight,
        },
        "text": {"stopwords": sorted(SPANISH_STOPWORDS), "stem_max_length": 7},
        "bm25": {
            "k1": bm25.k1,
            "b": bm25.b,
            "average_length": bm25.average_length,
            "idf": bm25.idf,
            "lengths": bm25.lengths,
            "term_frequencies": [dict(tf) for tf in bm25.term_frequencies],
        },
        "chunks": [
            {
                "source": chunk.source,
                "section": chunk.metadata.get("section"),
                "page": chunk.metadata.get("page"),
                "text": chunk.text,
            }
            for chunk in chunks
        ],
        "vectors": [[round(float(x), 6) for x in vector] for vector in vectors],
    }


def write_index(index: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def answer_from_ids(
    question: str,
    ids: list[int],
    chunks: list[Document],
    llm: LLM | None,
    organization: str = "la organización",
) -> Answer:
    """Genera la respuesta a partir de los fragmentos que el navegador ha recuperado.

    `ids` son las posiciones de los fragmentos en la colección, en orden de relevancia.
    """
    if any(not 0 <= i < len(chunks) for i in ids):
        raise ValueError("Número de fragmento fuera de rango")
    results = [
        SearchResult(document=chunks[i], score=0.0, rank=position)
        for position, i in enumerate(ids, start=1)
    ]
    return AnswerGenerator(llm, organization=organization).answer(question, results)


class GenerationService:
    """Lógica de la función de generación, independiente de la plataforma que la aloje.

    `handle` recibe el cuerpo JSON de la petición, la contraseña enviada por el
    navegador y un identificador del cliente, y devuelve `(código HTTP, respuesta)`.
    Con `{"check": true}` solo comprueba la contraseña (204 si es correcta).
    """

    def __init__(
        self,
        chunks: list[Document],
        llm: LLM | None,
        password: str,
        organization: str = "la organización",
        rate_limit: int = 20,
        max_fragments: int = 8,
    ) -> None:
        if not password:
            raise ValueError("Falta configurar FUNDAMENTA_PASSWORD")
        self.chunks = chunks
        self.llm = llm
        self.password = password.encode("utf-8")
        self.organization = organization
        self.max_fragments = max_fragments
        self.fingerprint = corpus_fingerprint(chunks)
        self.limiter = RateLimiter(rate_limit)

    @classmethod
    def from_environment(cls, root: str | Path = ".") -> GenerationService:
        """Configuración a partir de las variables de entorno (ver docs/despliegue.md)."""
        data = Path(root) / os.environ.get("FUNDAMENTA_DATA", "data/novatech")
        default_organization = "NovaTech" if data.name == "novatech" else "la organización"
        return cls(
            chunks=load_chunks(data),
            llm=get_llm(os.environ.get("FUNDAMENTA_LLM", "claude")),
            password=os.environ.get("FUNDAMENTA_PASSWORD", ""),
            organization=os.environ.get("FUNDAMENTA_ORGANIZATION", default_organization),
            rate_limit=int(os.environ.get("FUNDAMENTA_RATE_LIMIT", "20")),
        )

    def handle(self, payload: dict, password: str, client: str) -> tuple[int, dict | None]:
        # El límite se aplica antes que la contraseña para frenar los intentos de adivinarla
        if not self.limiter.allow(client):
            return 429, {"error": "Demasiadas preguntas seguidas. Espere un momento."}
        if not hmac.compare_digest(password.encode("utf-8"), self.password):
            return 401, {"error": "Contraseña incorrecta"}
        if payload.get("check"):
            return 204, None
        if payload.get("fingerprint") != self.fingerprint:
            return 409, {"error": "La documentación se ha actualizado. Recargue la página."}

        question = str(payload.get("question", "")).strip()
        if not question or len(question) > MAX_QUESTION_LENGTH:
            return 400, {
                "error": f"La pregunta debe tener entre 1 y {MAX_QUESTION_LENGTH} caracteres"
            }
        ids = payload.get("ids")
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= self.max_fragments
            or not all(type(i) is int and 0 <= i < len(self.chunks) for i in ids)
        ):
            return 400, {"error": "Lista de fragmentos no válida"}

        try:
            answer = answer_from_ids(question, ids, self.chunks, self.llm, self.organization)
        except LLMError as error:
            return 502, {"error": str(error)}
        return 200, answer_to_dict(answer)
