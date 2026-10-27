from pathlib import Path

import pytest

from fundamenta.loaders import (
    load_directory,
    load_document,
    load_markdown,
    load_pdf,
    load_text,
    normalize,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "novatech"


def test_normalize_cleans_line_endings_and_blank_lines():
    raw = "Hola   \r\nmundo\r\n\r\n\r\n\r\nadiós\n"
    assert normalize(raw) == "Hola\nmundo\n\nadiós"


def test_load_text(tmp_path):
    path = tmp_path / "nota.txt"
    # write_bytes para que Windows no traduzca los saltos de línea al escribir
    path.write_bytes("Primera línea  \r\nSegunda línea\r\n".encode())

    [doc] = load_text(path)

    assert doc.text == "Primera línea\nSegunda línea"
    assert doc.metadata == {"source": "nota.txt", "type": "txt"}


def test_load_markdown_extracts_title(tmp_path):
    path = tmp_path / "guia.md"
    path.write_text("# Guía rápida\n\nContenido.\n", encoding="utf-8")

    [doc] = load_markdown(path)

    assert doc.metadata["title"] == "Guía rápida"
    assert doc.metadata["type"] == "md"


def test_load_pdf_returns_one_document_per_page():
    docs = load_pdf(DATA_DIR / "beneficios-2026.pdf")

    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "1.200 €" in docs[0].text
    assert "Programa de referidos" in docs[1].text


def test_load_document_rejects_unknown_formats(tmp_path):
    path = tmp_path / "datos.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Formato no soportado"):
        load_document(path)


def test_load_directory_loads_the_novatech_dataset():
    docs = load_directory(DATA_DIR)

    sources = {d.source for d in docs}
    assert sources == {
        "beneficios-2026.pdf",
        "faq-soporte-it.txt",
        "manual-del-empleado.md",
        "politica-de-gastos-y-viajes.md",
        "productos-y-precios.md",
    }
    # 4 archivos de texto + 2 páginas del PDF
    assert len(docs) == 6
    assert all(d.text for d in docs)


def test_load_directory_ignores_unsupported_files(tmp_path):
    (tmp_path / "a.txt").write_text("texto", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,y", encoding="utf-8")

    docs = load_directory(tmp_path)

    assert [d.source for d in docs] == ["a.txt"]
