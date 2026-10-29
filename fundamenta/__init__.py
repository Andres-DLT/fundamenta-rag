"""Fundamenta: motor RAG construido desde cero.

Respuestas fundamentadas en la documentación de una organización, con citas
verificables. Cada módulo del paquete corresponde a un módulo del programa técnico
(carpeta `modulos/`).
"""

from fundamenta.chunking import chunk_documents
from fundamenta.embeddings import LSAEmbedder, SentenceTransformerEmbedder
from fundamenta.generation import Answer, AnswerGenerator
from fundamenta.llm import ClaudeLLM, OllamaLLM, OpenAICompatibleLLM
from fundamenta.loaders import Document, load_directory, load_document
from fundamenta.pipeline import PipelineConfig, RAGPipeline
from fundamenta.retriever import Retriever, SearchResult

__version__ = "1.1.0"

__all__ = [
    "Answer",
    "AnswerGenerator",
    "ClaudeLLM",
    "Document",
    "LSAEmbedder",
    "OllamaLLM",
    "OpenAICompatibleLLM",
    "PipelineConfig",
    "RAGPipeline",
    "Retriever",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "chunk_documents",
    "load_directory",
    "load_document",
]
