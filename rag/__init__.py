"""Build a RAG from Scratch: un sistema RAG construido pieza a pieza.

Cada módulo corresponde a un capítulo del curso (ver la carpeta `chapters/`).
"""

from rag.loaders import Document, load_directory, load_document

__version__ = "0.1.0"

__all__ = ["Document", "load_directory", "load_document"]
