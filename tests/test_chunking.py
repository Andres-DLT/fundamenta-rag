from pathlib import Path

import pytest

from fundamenta.chunking import (
    chunk_documents,
    chunk_fixed,
    chunk_recursive,
    split_markdown_sections,
    split_recursive,
)
from fundamenta.loaders import Document, load_directory

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "novatech"


# --- Estrategia 1: tamaño fijo -------------------------------------------------


def test_chunk_fixed_without_overlap_splits_into_consecutive_windows():
    assert chunk_fixed("abcdefghij", chunk_size=4) == ["abcd", "efgh", "ij"]


def test_chunk_fixed_with_overlap_repeats_the_end_of_each_chunk():
    chunks = chunk_fixed("abcdefghij", chunk_size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous[-1] == current[0]


def test_chunk_fixed_handles_short_and_empty_texts():
    assert chunk_fixed("abc", chunk_size=10, overlap=5) == ["abc"]
    assert chunk_fixed("", chunk_size=10) == []


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_chunk_fixed_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed("texto", chunk_size=chunk_size, overlap=overlap)


# --- Estrategia 2: recursiva ---------------------------------------------------


def test_split_recursive_keeps_all_the_text():
    text = "Primer párrafo.\n\nSegundo párrafo, algo más largo. Con dos frases.\nY una línea."

    pieces = split_recursive(text, chunk_size=20)

    assert "".join(pieces) == text
    assert all(len(piece) <= 20 for piece in pieces)


def test_split_recursive_falls_back_to_fixed_windows_for_long_words():
    pieces = split_recursive("a" * 25, chunk_size=10)

    assert pieces == ["a" * 10, "a" * 10, "a" * 5]


def test_chunk_recursive_prefers_paragraph_boundaries():
    first = "Este es el primer párrafo."
    second = "Y este es el segundo."

    chunks = chunk_recursive(f"{first}\n\n{second}", chunk_size=30, overlap=0)

    assert chunks == [first, second]


def test_chunk_recursive_respects_chunk_size_and_keeps_every_word():
    text = (DATA_DIR / "faq-soporte-it.txt").read_text(encoding="utf-8")

    chunks = chunk_recursive(text, chunk_size=200, overlap=0)

    assert all(len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chunk_recursive_overlaps_whole_sentences():
    text = "Uno dos. Tres cuatro. Cinco seis. Siete ocho."

    # Piezas: "Uno dos. " (9), "Tres cuatro. " (13), "Cinco seis. " (12), "Siete ocho." (11)
    chunks = chunk_recursive(text, chunk_size=25, overlap=13)

    assert chunks == [
        "Uno dos. Tres cuatro.",
        "Tres cuatro. Cinco seis.",
        "Cinco seis. Siete ocho.",
    ]


# --- Estrategia 3: secciones de Markdown --------------------------------------


def test_split_markdown_sections_tracks_the_heading_path():
    text = "Intro sin título.\n# A\ntexto a\n## B\ntexto b\n## C\ntexto c\n# D\ntexto d"

    sections = split_markdown_sections(text)

    assert [path for path, _ in sections] == ["", "A", "A > B", "A > C", "D"]
    assert sections[2][1] == "## B\ntexto b"


# --- Todo junto ----------------------------------------------------------------


def test_chunk_documents_keeps_metadata_and_adds_chunk_info():
    doc = Document(
        text="# Guía\n\nIntroducción.\n\n## Uno\n\nPrimera sección.\n\n## Dos\n\nSegunda sección.",
        metadata={"source": "guia.md", "type": "md"},
    )

    chunks = chunk_documents([doc], chunk_size=100, overlap=0)

    assert [c.metadata for c in chunks] == [
        {"source": "guia.md", "type": "md", "chunk": 0, "section": "Guía"},
        {"source": "guia.md", "type": "md", "chunk": 1, "section": "Guía > Uno"},
        {"source": "guia.md", "type": "md", "chunk": 2, "section": "Guía > Dos"},
    ]


def test_chunk_documents_drops_heading_only_chunks():
    doc = Document(
        text="## Permisos\n\n" + "Una tabla larga. " * 5,
        metadata={"source": "manual.md", "type": "md"},
    )

    chunks = chunk_documents([doc], chunk_size=90, overlap=0)

    assert [c.text.startswith("Una tabla") for c in chunks] == [True]
    assert chunks[0].metadata["section"] == "Permisos"
    assert chunks[0].metadata["chunk"] == 0


def test_chunk_documents_on_the_novatech_dataset():
    documents = load_directory(DATA_DIR)

    chunks = chunk_documents(documents, chunk_size=500, overlap=50)

    assert len(chunks) > len(documents)
    assert all(len(chunk.text) <= 500 for chunk in chunks)
    assert {c.source for c in chunks} == {d.source for d in documents}

    [vacations] = [c for c in chunks if "23 días laborables" in c.text]
    assert vacations.source == "manual-del-empleado.md"
    assert vacations.metadata["section"] == "Manual del empleado de NovaTech > Vacaciones"
