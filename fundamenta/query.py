"""Módulo 10: reescritura de consultas.

Las preguntas de los usuarios suelen ser coloquiales ("me han robado el laptop") y no
usan el vocabulario de la documentación ("pérdida o robo del portátil"). Un LLM puede
reformular la pregunta como una consulta de búsqueda más explícita antes de recuperar.

Cuesta una llamada adicional al LLM por pregunta, por lo que conviene activarlo solo si
la evaluación (módulo 09) muestra una mejora en su caso de uso.
"""

from __future__ import annotations

from fundamenta.llm import LLM

REWRITE_PROMPT = """Reformule la siguiente pregunta como una consulta para un buscador de
documentación interna de empresa. Use un lenguaje formal, sustituya expresiones
coloquiales por los términos que aparecerían en un documento oficial y añada sinónimos
relevantes. No responda a la pregunta.

Pregunta: {question}

Devuelva únicamente la consulta reformulada, en una sola línea."""


def rewrite_query(llm: LLM, question: str) -> str:
    """Devuelve la pregunta reformulada, o la original si el LLM no devuelve nada útil."""
    rewritten = llm.generate(
        "Usted reformula preguntas para un sistema de búsqueda documental.",
        REWRITE_PROMPT.format(question=question),
    )
    rewritten = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
    return rewritten or question
