"""Módulo 02: segmentación de documentos (chunking).

Divide los documentos en fragmentos (chunks) pequeños para que la búsqueda encuentre
el párrafo concreto que responde a la pregunta, y no un documento entero.

Todos los tamaños se miden en caracteres (regla práctica: 1 token ≈ 4 caracteres).
"""

from __future__ import annotations

import re

from fundamenta.loaders import Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _check_sizes(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size debe ser positivo (recibido: {chunk_size})")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap debe estar entre 0 y chunk_size - 1 (recibido: {overlap})")


def chunk_fixed(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Estrategia 1: ventanas de tamaño fijo.

    Cada chunk empieza `chunk_size - overlap` caracteres después del anterior, así que
    los últimos `overlap` caracteres de un chunk se repiten al principio del siguiente.
    Es la estrategia más simple, pero corta palabras y frases por la mitad.
    """
    _check_sizes(chunk_size, overlap)
    if not text:
        return []
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, max(len(text) - overlap, 1), step)]


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Como `str.split`, pero cada parte conserva su separador al final.

    Así, unir las partes con "" devuelve exactamente el texto original.
    """
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [piece for piece in pieces if piece]


def split_recursive(
    text: str, chunk_size: int, separators: tuple[str, ...] = DEFAULT_SEPARATORS
) -> list[str]:
    """Divide el texto en piezas de como máximo `chunk_size` caracteres.

    Intenta cortar por el separador más "grande" (párrafos). Las piezas que siguen
    siendo demasiado largas se dividen con el siguiente separador (líneas, frases,
    palabras) y, como último recurso, en ventanas de tamaño fijo.
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        return chunk_fixed(text, chunk_size)

    separator, remaining = separators[0], separators[1:]
    pieces = []
    for part in _split_keeping_separator(text, separator):
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(split_recursive(part, chunk_size, remaining))
    return pieces


def merge_pieces(pieces: list[str], chunk_size: int, overlap: int = 0) -> list[str]:
    """Junta piezas consecutivas en chunks de como máximo `chunk_size` caracteres.

    Al cerrar un chunk, el siguiente empieza con las últimas piezas del anterior que
    quepan en `overlap` caracteres. El solapamiento se hace por piezas completas
    (frases o palabras), nunca cortando una palabra.
    """
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) > chunk_size:
            chunks.append("".join(current))
            while current and (length > overlap or length + len(piece) > chunk_size):
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Estrategia 2: dividir por separadores naturales y volver a juntar.

    Los chunks terminan, siempre que se pueda, al final de un párrafo, una línea o
    una frase.
    """
    _check_sizes(chunk_size, overlap)
    return merge_pieces(split_recursive(text, chunk_size), chunk_size, overlap)


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Estrategia 3: respetar la estructura de un documento Markdown.

    Devuelve una lista de `(ruta, texto)`, donde `ruta` son los títulos que contienen
    a la sección, por ejemplo "Manual del empleado > Vacaciones". El texto de cada
    sección incluye su propio título.
    """
    sections = []
    path: list[str] = []
    current_path = ""
    start = 0
    for match in _HEADING.finditer(text):
        body = text[start : match.start()].strip()
        if body:
            sections.append((current_path, body))
        level = len(match.group(1))
        path = path[: level - 1] + [match.group(2)]
        current_path = " > ".join(path)
        start = match.start()
    body = text[start:].strip()
    if body:
        sections.append((current_path, body))
    return sections


def contextualize(chunk: Document) -> str:
    """Texto que se indexa para buscar un chunk: su sección seguida de su contenido.

    Así un chunk con una tabla de permisos sin la palabra "permiso" sigue siendo
    localizable, porque su sección ("... > Permisos retribuidos") sí la contiene.
    """
    section = chunk.metadata.get("section")
    return f"{section}\n\n{chunk.text}" if section else chunk.text


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, overlap: int = 50
) -> list[Document]:
    """Divide una lista de documentos en chunks, conservando sus metadatos.

    Los documentos Markdown se dividen primero por secciones y la ruta de títulos se
    guarda en `metadata["section"]`. Cada chunk lleva además su posición dentro del
    documento en `metadata["chunk"]`.

    Se descartan los chunks que solo contienen un título (pasa cuando el contenido de
    la sección no cabe en el mismo chunk): el título ya está en `metadata["section"]`.
    """
    _check_sizes(chunk_size, overlap)
    chunks = []
    for doc in documents:
        if doc.metadata.get("type") == "md":
            sections = split_markdown_sections(doc.text)
        else:
            sections = [("", doc.text)]

        index = 0
        for section, text in sections:
            for piece in chunk_recursive(text, chunk_size, overlap):
                if _HEADING.fullmatch(piece):
                    continue
                metadata = {**doc.metadata, "chunk": index}
                if section:
                    metadata["section"] = section
                chunks.append(Document(text=piece, metadata=metadata))
                index += 1
    return chunks
