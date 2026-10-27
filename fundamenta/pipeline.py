"""Módulo 08: el pipeline completo.

Une todas las piezas: ingesta -> segmentación -> indexación -> recuperación ->
generación. Es la interfaz principal de la librería:

    pipeline = RAGPipeline.from_directory("data/novatech", embedder=LSAEmbedder())
    answer = pipeline.ask("¿Cuántos días de vacaciones tengo?")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fundamenta.chunking import chunk_documents
from fundamenta.embeddings import Embedder
from fundamenta.generation import Answer, AnswerGenerator
from fundamenta.llm import LLM
from fundamenta.loaders import Document, load_directory
from fundamenta.query import rewrite_query
from fundamenta.retriever import Mode, Retriever, SearchResult


@dataclass
class PipelineConfig:
    """Parámetros del pipeline. Los valores por defecto son los validados en el módulo 09."""

    chunk_size: int = 500
    overlap: int = 50
    k: int = 4
    mode: Mode = "hybrid"
    contextual: bool = True
    lexical_weight: float = 0.5
    rewrite_queries: bool = False
    organization: str = "la organización"


class RAGPipeline:
    """Sistema RAG completo sobre una colección de documentos."""

    def __init__(
        self,
        chunks: list[Document],
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.chunks = chunks
        self.llm = llm
        # Sin embedder solo es posible la búsqueda léxica
        mode = self.config.mode if embedder is not None else "lexical"
        self.retriever = Retriever(
            chunks,
            embedder=embedder,
            mode=mode,
            contextual=self.config.contextual,
            lexical_weight=self.config.lexical_weight,
        )
        self.generator = AnswerGenerator(llm, organization=self.config.organization)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: PipelineConfig | None = None,
    ) -> RAGPipeline:
        config = config or PipelineConfig()
        chunks = chunk_documents(documents, config.chunk_size, config.overlap)
        return cls(chunks, embedder=embedder, llm=llm, config=config)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: PipelineConfig | None = None,
    ) -> RAGPipeline:
        """Carga, segmenta e indexa todos los documentos soportados de una carpeta."""
        return cls.from_documents(load_directory(directory), embedder, llm, config)

    @property
    def mode(self) -> str:
        return self.retriever.mode

    def search(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Recupera los fragmentos más relevantes sin generar respuesta.

        Con `rewrite_queries=True` y un LLM configurado, la pregunta se reformula antes
        de buscar (módulo 10).
        """
        query = question
        if self.config.rewrite_queries and self.llm is not None:
            query = rewrite_query(self.llm, question)
        return self.retriever.search(query, k=k or self.config.k, where=where)

    def ask(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> Answer:
        """Responde a una pregunta citando los fragmentos que la respaldan."""
        return self.generator.answer(question, self.search(question, k=k, where=where))
