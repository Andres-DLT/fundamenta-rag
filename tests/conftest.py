from pathlib import Path

import pytest

from fundamenta.chunking import chunk_documents
from fundamenta.loaders import Document, load_directory

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "novatech"
QUESTIONS = ROOT / "evaluacion" / "preguntas.jsonl"


class FakeLLM:
    """LLM de prueba: devuelve una respuesta fija y registra las llamadas."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture(scope="session")
def novatech_chunks() -> list[Document]:
    return chunk_documents(load_directory(DATA_DIR))
