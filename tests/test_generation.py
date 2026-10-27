"""Tests del módulo 07: proveedores de LLM y generación con citas."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from fundamenta.generation import (
    NO_ANSWER,
    AnswerGenerator,
    build_prompt,
    cited_numbers,
    to_sources,
)
from fundamenta.llm import ClaudeLLM, LLMError, OllamaLLM, get_llm
from fundamenta.loaders import Document
from fundamenta.retriever import SearchResult
from tests.conftest import FakeLLM


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            Document(
                "Cada persona tiene 23 días laborables de vacaciones.",
                {"source": "manual.md", "section": "Manual > Vacaciones", "chunk": 5},
            ),
            0.9,
            1,
        ),
        SearchResult(
            Document("Beneficios de la empresa.", {"source": "beneficios.pdf", "page": 2}),
            0.4,
            2,
        ),
    ]


def test_source_labels():
    first, second = to_sources(_results())

    assert first.label == "manual.md › Vacaciones"
    assert second.label == "beneficios.pdf › página 2"


def test_build_prompt_numbers_the_fragments():
    prompt = build_prompt("¿Vacaciones?", to_sources(_results()))

    assert "[1] Fuente: manual.md › Vacaciones" in prompt
    assert "[2] Fuente: beneficios.pdf › página 2" in prompt
    assert prompt.endswith("Pregunta: ¿Vacaciones?")


def test_cited_numbers():
    assert cited_numbers("Tienes 23 días [1][3]. Ver también [1].") == {1, 3}


def test_generator_keeps_only_cited_sources():
    llm = FakeLLM("Tiene 23 días laborables de vacaciones [1].")

    answer = AnswerGenerator(llm, organization="NovaTech").answer("¿Vacaciones?", _results())

    assert answer.answered and answer.mode == "generative"
    assert [s.number for s in answer.sources] == [1]
    system, _ = llm.calls[0]
    assert "NovaTech" in system and NO_ANSWER in system


def test_generator_does_not_call_the_llm_without_results():
    llm = FakeLLM("no debería llamarse")

    answer = AnswerGenerator(llm).answer("¿Algo?", [])

    assert answer.text == NO_ANSWER and not answer.answered
    assert llm.calls == []


def test_generator_detects_abstention():
    answer = AnswerGenerator(FakeLLM(NO_ANSWER)).answer("¿Salario medio?", _results())

    assert not answer.answered
    assert answer.sources == []


def test_extractive_mode_without_llm():
    answer = AnswerGenerator().answer("¿Vacaciones?", _results())

    assert answer.mode == "extractive"
    assert "[1] manual.md › Vacaciones" in answer.text
    assert len(answer.sources) == 2


def test_get_llm():
    assert get_llm(None) is None
    assert get_llm("none") is None
    assert get_llm("ollama:llama3.1:8b").model == "llama3.1:8b"
    with pytest.raises(ValueError):
        get_llm("otro")


def test_ollama_llm_against_a_local_server():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"message": {"role": "assistant", "content": " Hola [1] "}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        llm = OllamaLLM(model="test", host=f"http://127.0.0.1:{server.server_address[1]}")
        assert llm.generate("sistema", "pregunta") == "Hola [1]"
    finally:
        thread.join(timeout=5)
        server.server_close()

    assert received["path"] == "/api/chat"
    assert received["body"]["model"] == "test"
    assert received["body"]["stream"] is False
    assert [m["role"] for m in received["body"]["messages"]] == ["system", "user"]


def test_ollama_llm_reports_connection_errors():
    llm = OllamaLLM(host="http://127.0.0.1:9", timeout=2)

    with pytest.raises(LLMError):
        llm.generate("sistema", "pregunta")


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _claude_with_response(monkeypatch, response):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-prueba")
    llm = ClaudeLLM()
    messages = _FakeMessages(response)
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return llm, messages


def test_claude_llm_request_and_text_extraction(monkeypatch):
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Tiene 23 días [1]."),
        ],
    )
    llm, messages = _claude_with_response(monkeypatch, response)

    assert llm.generate("sistema", "pregunta") == "Tiene 23 días [1]."
    assert messages.kwargs["model"] == "claude-opus-5-5"
    assert messages.kwargs["system"] == "sistema"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "pregunta"}]
    assert messages.kwargs["fallbacks"] == "default"
    assert messages.kwargs["output_config"] == {"effort": "medium"}


def test_claude_llm_raises_on_refusal(monkeypatch):
    llm, _ = _claude_with_response(monkeypatch, SimpleNamespace(stop_reason="refusal", content=[]))

    with pytest.raises(LLMError):
        llm.generate("sistema", "pregunta")


def test_rewrite_query():
    from fundamenta.query import rewrite_query

    llm = FakeLLM("pérdida o robo del portátil corporativo\nlínea extra")
    assert (
        rewrite_query(llm, "me han robado el laptop") == "pérdida o robo del portátil corporativo"
    )
    assert rewrite_query(FakeLLM("  "), "pregunta original") == "pregunta original"


def test_pipeline_rewrites_queries_when_enabled():
    from fundamenta.pipeline import PipelineConfig, RAGPipeline
    from tests.conftest import DATA_DIR

    llm = FakeLLM("vacaciones días laborables")
    config = PipelineConfig(rewrite_queries=True)
    pipeline = RAGPipeline.from_directory(DATA_DIR, llm=llm, config=config)

    results = pipeline.search("¿cuánto descanso tengo?", k=1)

    assert "23 días laborables" in results[0].document.text
    assert "¿cuánto descanso tengo?" in llm.calls[0][1]
