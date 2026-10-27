"""Capítulo 01: cargar documentos.

Convierte archivos (.txt, .md y .pdf) en objetos `Document` con su texto y sus
metadatos. Los metadatos (archivo de origen, página...) son los que más adelante nos
permitirán citar de dónde sale cada respuesta.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass
class Document:
    """Un texto junto con la información de dónde viene."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "desconocido")

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Document(source={self.source!r}, chars={len(self.text)}, text={preview!r}...)"


def normalize(text: str) -> str:
    """Limpieza mínima del texto.

    - Unifica los saltos de línea (Windows usa \\r\\n).
    - Quita los espacios al final de cada línea.
    - Deja como máximo una línea en blanco seguida.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text(path: str | Path) -> list[Document]:
    """Carga un archivo de texto plano como un único documento."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return [Document(text=normalize(text), metadata={"source": path.name, "type": "txt"})]


def load_markdown(path: str | Path) -> list[Document]:
    """Carga un archivo Markdown como un único documento.

    Si el archivo empieza con un título (`# Título`), lo guardamos en los metadatos.
    """
    path = Path(path)
    text = normalize(path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {"source": path.name, "type": "md"}
    title = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if title:
        metadata["title"] = title.group(1).strip()
    return [Document(text=text, metadata=metadata)]


def load_pdf(path: str | Path) -> list[Document]:
    """Carga un PDF devolviendo un documento por página.

    Separar por páginas permite citar "beneficios-2026.pdf, página 2" en las respuestas.
    Las páginas sin texto (por ejemplo, escaneadas sin OCR) se descartan.
    """
    path = Path(path)
    reader = PdfReader(path)
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if text:
            metadata = {"source": path.name, "type": "pdf", "page": number}
            documents.append(Document(text=text, metadata=metadata))
    return documents


LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: str | Path) -> list[Document]:
    """Elige el loader adecuado según la extensión del archivo."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"Formato no soportado: {path.name} (soportados: {supported})")
    return loader(path)


def load_directory(directory: str | Path, recursive: bool = True) -> list[Document]:
    """Carga todos los archivos soportados de una carpeta, en orden alfabético.

    Los archivos con extensiones desconocidas se ignoran.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"No existe la carpeta: {directory}")
    pattern = "**/*" if recursive else "*"
    documents = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in LOADERS:
            documents.extend(load_document(path))
    return documents
