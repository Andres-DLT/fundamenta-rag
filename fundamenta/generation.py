"""Módulo 07: generación de respuestas con citas.

Construye el prompt con los fragmentos recuperados (numerados para poder citarlos),
llama al LLM y devuelve la respuesta junto con las fuentes que la respaldan.

Si no hay LLM configurado, funciona en modo extractivo: devuelve los fragmentos más
relevantes tal cual, con sus referencias. Es útil para demostraciones sin credenciales
y como alternativa cuando el proveedor no está disponible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fundamenta.llm import LLM
from fundamenta.retriever import SearchResult

NO_ANSWER = (
    "No dispongo de información suficiente en la documentación disponible para responder "
    "a esta pregunta."
)

SYSTEM_PROMPT = """Usted es el asistente de conocimiento interno de {organization}.

Responda a la pregunta utilizando exclusivamente la información de los fragmentos de
documentación que se le proporcionan.

Normas:
- Cite cada afirmación con el número del fragmento entre corchetes, por ejemplo [1] o [2][3].
- No utilice conocimientos externos ni haga suposiciones sobre lo que no aparece en los fragmentos.
- Si los fragmentos no contienen la respuesta, responda exactamente: "{no_answer}"
- Responda en el idioma de la pregunta, de forma clara y concisa."""


@dataclass
class Source:
    """Referencia a un fragmento de la documentación utilizado en la respuesta."""

    number: int
    source: str
    section: str | None
    page: int | None
    text: str
    score: float

    @property
    def label(self) -> str:
        """Referencia legible, por ejemplo "manual-del-empleado.md › Vacaciones"."""
        parts = [self.source]
        if self.section:
            parts.append(self.section.split(" > ")[-1])
        if self.page:
            parts.append(f"página {self.page}")
        return " › ".join(parts)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "generative"

    @property
    def answered(self) -> bool:
        return self.text != NO_ANSWER


def to_sources(results: list[SearchResult]) -> list[Source]:
    return [
        Source(
            number=position,
            source=result.document.source,
            section=result.document.metadata.get("section"),
            page=result.document.metadata.get("page"),
            text=result.document.text,
            score=result.score,
        )
        for position, result in enumerate(results, start=1)
    ]


def build_prompt(question: str, sources: list[Source]) -> str:
    """Prompt con los fragmentos numerados seguidos de la pregunta."""
    fragments = "\n\n".join(
        f"[{source.number}] Fuente: {source.label}\n{source.text}" for source in sources
    )
    return f"Fragmentos de documentación:\n\n{fragments}\n\nPregunta: {question}"


def cited_numbers(text: str) -> set[int]:
    """Números de fragmento citados en una respuesta: "[1][3]" -> {1, 3}."""
    return {int(number) for number in re.findall(r"\[(\d+)\]", text)}


class AnswerGenerator:
    """Genera respuestas fundamentadas a partir de los resultados de la búsqueda."""

    def __init__(self, llm: LLM | None = None, organization: str = "la organización") -> None:
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT.format(organization=organization, no_answer=NO_ANSWER)

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        sources = to_sources(results)
        if not sources:
            return Answer(question=question, text=NO_ANSWER, mode="none")
        if self.llm is None:
            return self._extractive(question, sources)

        text = self.llm.generate(self.system_prompt, build_prompt(question, sources))
        if NO_ANSWER in text:
            return Answer(question=question, text=NO_ANSWER, mode="generative")
        cited = cited_numbers(text)
        used = [source for source in sources if source.number in cited] or sources
        return Answer(question=question, text=text, sources=used, mode="generative")

    @staticmethod
    def _extractive(question: str, sources: list[Source]) -> Answer:
        lines = ["Fragmentos más relevantes de la documentación:"]
        lines += [f"[{s.number}] {s.label}\n{s.text}" for s in sources]
        return Answer(
            question=question, text="\n\n".join(lines), sources=sources, mode="extractive"
        )
