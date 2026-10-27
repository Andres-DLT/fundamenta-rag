"""Tests de los módulos 08 y 09: pipeline, servidor web, CLI y evaluación."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from fundamenta.cli import main
from fundamenta.embeddings import LSAEmbedder
from fundamenta.evaluation import (
    EvalQuestion,
    answer_is_correct,
    evaluate_answers,
    evaluate_retrieval,
    is_relevant,
    judge_faithfulness,
    load_questions,
)
from fundamenta.generation import NO_ANSWER, Answer
from fundamenta.loaders import Document
from fundamenta.pipeline import PipelineConfig, RAGPipeline
from fundamenta.server import create_server
from tests.conftest import DATA_DIR, QUESTIONS, FakeLLM

# --- Módulo 08: pipeline --------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    config = PipelineConfig(organization="NovaTech")
    return RAGPipeline.from_directory(DATA_DIR, embedder=LSAEmbedder(), config=config)


def test_pipeline_without_embedder_falls_back_to_lexical():
    pipeline = RAGPipeline.from_directory(DATA_DIR)

    assert pipeline.mode == "lexical"
    assert "23 días laborables" in pipeline.search("vacaciones al año", k=1)[0].document.text


def test_pipeline_ask_extractive(pipeline):
    answer = pipeline.ask("¿Cuántos días de vacaciones tengo?")

    assert answer.mode == "extractive"
    assert any("23 días laborables" in source.text for source in answer.sources)


def test_pipeline_ask_with_llm():
    llm = FakeLLM("Tiene 23 días laborables [1].")
    pipeline = RAGPipeline.from_directory(DATA_DIR, llm=llm)

    answer = pipeline.ask("¿Cuántos días de vacaciones tengo?")

    assert answer.text == "Tiene 23 días laborables [1]."
    assert len(answer.sources) == 1
    assert "23 días laborables" in llm.calls[0][1]


# --- Módulo 08: servidor web ----------------------------------------------------


@pytest.fixture(scope="module")
def server_url(pipeline):
    server = create_server(pipeline, port=0, examples=["¿Vacaciones?"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.mark.parametrize("path", ["/", "/index.html", "/?q=vacaciones"])
def test_server_serves_the_interface(server_url, path):
    with urllib.request.urlopen(server_url + path, timeout=10) as response:
        assert response.status == 200
        assert "Fundamenta" in response.read().decode("utf-8")


def test_server_health(server_url):
    with urllib.request.urlopen(server_url + "/api/health", timeout=10) as response:
        health = json.loads(response.read())

    assert health["status"] == "ok"
    assert health["organization"] == "NovaTech"
    assert health["examples"] == ["¿Vacaciones?"]


def test_server_answers_questions(server_url):
    status, body = _post(server_url + "/api/ask", {"question": "¿Cuántos días de vacaciones?"})

    assert status == 200
    assert body["mode"] == "extractive"
    assert body["sources"][0]["label"]


def test_server_rejects_invalid_questions(server_url):
    assert _post(server_url + "/api/ask", {"question": "  "})[0] == 400
    assert _post(server_url + "/api/ask", {"question": "x" * 2001})[0] == 400
    assert _post(server_url + "/api/otra", {})[0] == 404


# --- Módulo 08: CLI ---------------------------------------------------------------


def test_cli_search(capsys):
    main(["search", "dieta diaria España", "--data", str(DATA_DIR), "--k", "2"])

    output = capsys.readouterr().out
    assert output.startswith("1. ")
    assert "politica-de-gastos-y-viajes.md" in output


def test_cli_ask_extractive(capsys):
    main(["ask", "¿Cuántos días de vacaciones tengo?", "--data", str(DATA_DIR)])

    assert "23 días laborables" in capsys.readouterr().out


def test_cli_eval(capsys):
    main(["eval", "--data", str(DATA_DIR), "--questions", str(QUESTIONS), "--embedder", "none"])

    output = capsys.readouterr().out
    assert "| Léxica (BM25) |" in output


# --- Módulo 09: evaluación --------------------------------------------------------


def test_load_questions():
    questions = load_questions(QUESTIONS)

    assert len(questions) == 30
    assert {q.category for q in questions} == {"directa", "parafraseada", "sin_respuesta"}
    assert sum(not q.answerable for q in questions) == 3


def test_every_question_has_its_evidence_in_the_dataset(novatech_chunks):
    for question in load_questions(QUESTIONS):
        if question.answerable:
            assert any(is_relevant(chunk, question) for chunk in novatech_chunks), question.id


def test_is_relevant_ignores_accents_case_and_line_breaks():
    question = EvalQuestion("x", "?", "directa", "a.md", ["Revisión salarial"])

    assert is_relevant(Document("la revision\nsalarial", {"source": "a.md"}), question)
    assert not is_relevant(Document("la revision salarial", {"source": "b.md"}), question)


def test_evaluate_retrieval_metrics(pipeline):
    questions = [
        EvalQuestion("a", "vacaciones al año", "directa", "manual-del-empleado.md", ["23 días"]),
        EvalQuestion("b", "xyz", "directa", "inexistente.md", ["nada"]),
        EvalQuestion("c", "¿Pregunta sin respuesta?", "sin_respuesta"),
    ]

    report = evaluate_retrieval(lambda q, k: pipeline.search(q, k=k), questions, k=4)

    assert report.questions == 2
    assert report.hit_rate == 0.5
    assert 0 < report.mrr <= 0.5
    assert report.misses == ["b"]


def test_answer_is_correct():
    answerable = EvalQuestion("a", "?", "directa", "a.md", ["x"], ["23", "días"])
    unanswerable = EvalQuestion("n", "?", "sin_respuesta")

    assert answer_is_correct(Answer("?", "Son 23 DIAS [1]."), answerable)
    assert not answer_is_correct(Answer("?", "Son 22 días."), answerable)
    assert answer_is_correct(Answer("?", NO_ANSWER), unanswerable)
    assert not answer_is_correct(Answer("?", "Unos 30.000 €."), unanswerable)


def test_evaluate_answers():
    questions = [
        EvalQuestion("a", "¿Vacaciones?", "directa", "a.md", ["x"], ["23"]),
        EvalQuestion("n", "¿Salario?", "sin_respuesta"),
    ]
    replies = {"¿Vacaciones?": "Son 23 días [1].", "¿Salario?": NO_ANSWER}

    report = evaluate_answers(lambda q: Answer(q, replies[q]), questions)

    assert report.accuracy == 1.0
    assert report.abstention_accuracy == 1.0


def test_judge_faithfulness():
    answer = Answer("¿Vacaciones?", "Son 23 días [1].")

    assert judge_faithfulness(FakeLLM("SÍ"), answer)
    assert not judge_faithfulness(FakeLLM("NO"), answer)
    assert judge_faithfulness(FakeLLM("NO"), Answer("?", NO_ANSWER))
