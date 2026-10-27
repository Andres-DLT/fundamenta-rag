"""Módulo 08: interfaz de línea de comandos.

fundamenta ask "¿Cuántos días de vacaciones tengo?"
fundamenta search "dietas en viajes" --k 3
fundamenta serve --port 8000
fundamenta eval --embedder semantic

Las opciones principales también se pueden fijar con variables de entorno, lo que
facilita el despliegue en contenedores: FUNDAMENTA_DATA, FUNDAMENTA_EMBEDDER,
FUNDAMENTA_LLM, FUNDAMENTA_PASSWORD, FUNDAMENTA_RATE_LIMIT y PORT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fundamenta.embeddings import get_embedder
from fundamenta.evaluation import evaluate_answers, evaluate_retrieval, load_questions
from fundamenta.llm import get_llm
from fundamenta.pipeline import PipelineConfig, RAGPipeline

DEMO_EXAMPLES = [
    "¿Cuántos días de vacaciones tengo al año?",
    "Me han robado el portátil, ¿qué tengo que hacer?",
    "¿Cuánto me pagan de dieta si viajo a Lisboa?",
    "¿Qué incluye el plan Enterprise de NovaDesk?",
]


def build_pipeline(args: argparse.Namespace, mode: str | None = None) -> RAGPipeline:
    embedder = None if args.embedder == "none" else get_embedder(args.embedder)
    organization = args.organization or (
        "NovaTech" if Path(args.data).name == "novatech" else "la organización"
    )
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        k=args.k,
        mode=mode or args.mode,
        organization=organization,
    )
    return RAGPipeline.from_directory(args.data, embedder, get_llm(args.llm), config)


def cmd_ask(args: argparse.Namespace) -> None:
    answer = build_pipeline(args).ask(args.question)
    print(answer.text)
    if answer.sources and answer.mode == "generative":
        print("\nFuentes:")
        for source in answer.sources:
            print(f"  [{source.number}] {source.label}")


def cmd_search(args: argparse.Namespace) -> None:
    for result in build_pipeline(args).search(args.query):
        doc = result.document
        label = doc.metadata.get("section") or doc.source
        print(f"{result.rank}. ({result.score:.4f}) {doc.source} | {label}")
        print("   " + doc.text[:200].replace("\n", " ") + ("…" if len(doc.text) > 200 else ""))


def cmd_serve(args: argparse.Namespace) -> None:
    from fundamenta.server import serve

    pipeline = build_pipeline(args)
    examples = DEMO_EXAMPLES if Path(args.data).name == "novatech" else []
    serve(
        pipeline,
        host=args.host,
        port=args.port,
        examples=examples,
        password=args.password or None,
        rate_limit=args.rate_limit,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    modes = ["lexical"] if args.embedder == "none" else ["lexical", "semantic", "hybrid"]
    names = {
        "lexical": "Léxica (BM25)",
        "semantic": f"Semántica ({args.embedder})",
        "hybrid": f"Híbrida (BM25 + {args.embedder})",
    }
    answerable = sum(q.answerable for q in questions)
    print(f"Evaluación de recuperación: {answerable} preguntas, k={args.k}\n")
    print("| Configuración | Acierto@k | MRR | Directas | Parafraseadas |")
    print("|---|---|---|---|---|")
    for mode in modes:
        pipeline = build_pipeline(args, mode=mode)
        report = evaluate_retrieval(pipeline.search, questions, k=args.k)
        cat = report.by_category
        print(
            f"| {names[mode]} | {report.hit_rate:.0%} | {report.mrr:.3f} "
            f"| {cat.get('directa', 0):.0%} | {cat.get('parafraseada', 0):.0%} |"
        )
        if report.misses:
            print(f"|   fallos: {', '.join(report.misses)} | | | | |")

    if args.answers:
        if args.llm in (None, "none"):
            sys.exit("La evaluación de respuestas requiere un LLM (--llm claude u --llm ollama)")
        report = evaluate_answers(build_pipeline(args).ask, questions)
        print(f"\nRespuestas correctas: {report.accuracy:.0%} ({report.questions} preguntas)")
        print(f"Abstención correcta en preguntas sin respuesta: {report.abstention_accuracy:.0%}")
        if report.failures:
            print(f"Fallos: {', '.join(report.failures)}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fundamenta",
        description="Motor RAG: respuestas fundamentadas en su documentación, con citas.",
    )
    common = argparse.ArgumentParser(add_help=False)
    env = os.environ.get
    common.add_argument(
        "--data", default=env("FUNDAMENTA_DATA", "data/novatech"), help="carpeta de documentos"
    )
    common.add_argument(
        "--embedder",
        default=env("FUNDAMENTA_EMBEDDER", "lsa"),
        choices=["lsa", "semantic", "none"],
        help="lsa: sin descargas; semantic: modelo multilingüe (extra [semantic])",
    )
    common.add_argument(
        "--llm",
        default=env("FUNDAMENTA_LLM", "none"),
        help="none (extractivo), claude[:modelo] u ollama[:modelo]",
    )
    common.add_argument("--mode", default="hybrid", choices=["lexical", "semantic", "hybrid"])
    common.add_argument("--k", type=int, default=4, help="fragmentos a recuperar")
    common.add_argument("--chunk-size", type=int, default=500)
    common.add_argument("--overlap", type=int, default=50)
    common.add_argument("--organization", help="nombre de la organización para el asistente")

    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", parents=[common], help="responder a una pregunta")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    search = commands.add_parser("search", parents=[common], help="buscar fragmentos")
    search.add_argument("query")
    search.set_defaults(func=cmd_search)

    serve = commands.add_parser("serve", parents=[common], help="arrancar la demo web")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=int(env("PORT", "8000")))
    serve.add_argument(
        "--password",
        default=env("FUNDAMENTA_PASSWORD"),
        help="contraseña de acceso (recomendado si el servidor es público)",
    )
    serve.add_argument(
        "--rate-limit",
        type=int,
        default=int(env("FUNDAMENTA_RATE_LIMIT", "30")),
        help="preguntas por minuto y cliente",
    )
    serve.set_defaults(func=cmd_serve)

    evaluate = commands.add_parser("eval", parents=[common], help="evaluar la calidad")
    evaluate.add_argument("--questions", default="evaluacion/preguntas.jsonl")
    evaluate.add_argument(
        "--answers", action="store_true", help="evaluar también las respuestas (usa el LLM)"
    )
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = make_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
