"""Módulo 09: evaluación.

Mide la calidad del sistema con un conjunto de preguntas de referencia, en dos niveles:

1. Recuperación: ¿está el fragmento que contiene la respuesta entre los `k` primeros?
   - Tasa de acierto (hit rate o recall@k): proporción de preguntas en las que sí está.
   - MRR (Mean Reciprocal Rank): media de 1/posición del primer fragmento relevante.
     Premia que el fragmento correcto aparezca arriba, no solo que aparezca.
2. Respuesta (requiere un LLM): ¿contiene la respuesta los datos esperados? ¿Se abstiene
   el sistema cuando la documentación no contiene la respuesta?

Un fragmento se considera relevante si procede del documento esperado y contiene todos
los textos de evidencia. Este criterio no depende de cómo se haya segmentado el
documento, por lo que permite comparar configuraciones distintas de forma justa.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fundamenta.generation import NO_ANSWER, Answer
from fundamenta.llm import LLM
from fundamenta.loaders import Document
from fundamenta.retriever import SearchResult
from fundamenta.text import strip_accents


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str
    source: str | None = None
    evidence: list[str] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Carga un archivo JSONL con una pregunta por línea."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalQuestion(**json.loads(line)) for line in lines if line.strip()]


def _normalize(text: str) -> str:
    return " ".join(strip_accents(text.lower()).split())


def is_relevant(document: Document, question: EvalQuestion) -> bool:
    text = _normalize(document.text)
    return document.source == question.source and all(
        _normalize(evidence) in text for evidence in question.evidence
    )


@dataclass
class RetrievalReport:
    k: int
    questions: int
    hit_rate: float
    mrr: float
    by_category: dict[str, float]
    misses: list[str]


def evaluate_retrieval(
    search: Callable[[str, int], list[SearchResult]],
    questions: list[EvalQuestion],
    k: int = 4,
) -> RetrievalReport:
    """Evalúa una función de búsqueda `search(pregunta, k)` sobre las preguntas con respuesta."""
    answerable = [q for q in questions if q.answerable]
    reciprocal_ranks: list[float] = []
    hits_by_category: dict[str, list[bool]] = {}
    misses = []
    for question in answerable:
        results = search(question.question, k)
        rank = next(
            (r.rank for r in results if is_relevant(r.document, question)),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        hits_by_category.setdefault(question.category, []).append(rank is not None)
        if rank is None:
            misses.append(question.id)

    n = len(answerable) or 1
    return RetrievalReport(
        k=k,
        questions=len(answerable),
        hit_rate=sum(rr > 0 for rr in reciprocal_ranks) / n,
        mrr=sum(reciprocal_ranks) / n,
        by_category={cat: sum(hits) / len(hits) for cat, hits in hits_by_category.items()},
        misses=misses,
    )


def answer_is_correct(answer: Answer, question: EvalQuestion) -> bool:
    """Comprueba una respuesta generada frente a lo esperado.

    - Preguntas con respuesta: deben aparecer todos los textos de `answer_contains`.
    - Preguntas sin respuesta: el sistema debe abstenerse.
    """
    if not question.answerable:
        return not answer.answered
    text = _normalize(answer.text)
    return answer.answered and all(_normalize(term) in text for term in question.answer_contains)


@dataclass
class AnswerReport:
    questions: int
    accuracy: float
    abstention_accuracy: float
    failures: list[str]


def evaluate_answers(ask: Callable[[str], Answer], questions: list[EvalQuestion]) -> AnswerReport:
    """Evalúa las respuestas generadas. Cada pregunta supone una llamada al LLM."""
    correct, failures = [], []
    abstentions = []
    for question in questions:
        ok = answer_is_correct(ask(question.question), question)
        correct.append(ok)
        if not question.answerable:
            abstentions.append(ok)
        if not ok:
            failures.append(question.id)
    return AnswerReport(
        questions=len(questions),
        accuracy=sum(correct) / (len(correct) or 1),
        abstention_accuracy=sum(abstentions) / (len(abstentions) or 1),
        failures=failures,
    )


JUDGE_PROMPT = """Evalúe si la RESPUESTA está completamente respaldada por los FRAGMENTOS.
Una respuesta está respaldada si cada dato que afirma aparece en los fragmentos.
Si la respuesta indica que no hay información suficiente, considérela respaldada.

FRAGMENTOS:
{fragments}

PREGUNTA: {question}

RESPUESTA: {answer}

Conteste únicamente con una palabra: SI o NO."""


def judge_faithfulness(llm: LLM, answer: Answer) -> bool:
    """LLM como evaluador: ¿la respuesta se apoya solo en las fuentes citadas?

    Detecta alucinaciones que una comprobación por palabras clave no ve. Conviene
    usar un modelo distinto, o al menos un prompt distinto, del que generó la respuesta.
    """
    if answer.text == NO_ANSWER:
        return True
    fragments = "\n\n".join(f"[{s.number}] {s.text}" for s in answer.sources)
    verdict = llm.generate(
        "Usted es un evaluador riguroso de sistemas de búsqueda documental.",
        JUDGE_PROMPT.format(fragments=fragments, question=answer.question, answer=answer.text),
    )
    return _normalize(verdict).startswith("si")
