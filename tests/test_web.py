"""Tests de la versión web: índice para el navegador y función de generación."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from fundamenta.embeddings import LSAEmbedder
from fundamenta.generation import NO_ANSWER
from fundamenta.llm import LLMError
from fundamenta.pipeline import PipelineConfig
from fundamenta.web import (
    GenerationService,
    answer_from_ids,
    build_index,
    corpus_fingerprint,
    load_chunks,
)
from tests.conftest import DATA_DIR, ROOT, FakeLLM


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(DATA_DIR)


def test_load_chunks_matches_the_pipeline(chunks, novatech_chunks):
    assert [c.text for c in chunks] == [c.text for c in novatech_chunks]


def test_fingerprint_detects_changes(chunks):
    assert corpus_fingerprint(chunks) == corpus_fingerprint(load_chunks(DATA_DIR))
    assert corpus_fingerprint(chunks) != corpus_fingerprint(chunks[1:])


def test_build_index_contains_everything_the_browser_needs(chunks):
    index = build_index(chunks, LSAEmbedder(), PipelineConfig(organization="NovaTech"), ["¿Hola?"])

    assert index["organization"] == "NovaTech"
    assert index["examples"] == ["¿Hola?"]
    assert len(index["chunks"]) == len(index["vectors"]) == len(index["bm25"]["lengths"])
    assert index["chunks"][0]["source"] == chunks[0].source
    assert "vacacio" in index["bm25"]["idf"]
    assert index["retrieval"]["lexical_weight"] == 0.5
    assert "de" in index["text"]["stopwords"]
    json.dumps(index)  # serializable


def test_index_vectors_are_normalized(chunks):
    index = build_index(chunks, LSAEmbedder())
    norms = np.linalg.norm(np.array(index["vectors"]), axis=1)
    assert np.allclose(norms[norms > 0], 1, atol=1e-4)


def test_answer_from_ids_cites_the_selected_fragments(chunks):
    llm = FakeLLM("Tiene 23 días laborables [1].")

    answer = answer_from_ids("¿Vacaciones?", [17, 4], chunks, llm, "NovaTech")

    assert answer.mode == "generative"
    assert [s.number for s in answer.sources] == [1]
    assert answer.sources[0].text == chunks[17].text
    system, prompt = llm.calls[0]
    assert "NovaTech" in system
    assert chunks[4].text in prompt


def test_answer_from_ids_rejects_unknown_fragments(chunks):
    with pytest.raises(ValueError):
        answer_from_ids("¿Vacaciones?", [len(chunks)], chunks, FakeLLM("x"))


# --- Servicio de generación -----------------------------------------------------


def make_service(chunks, reply="Respuesta [1].", **kwargs):
    return GenerationService(chunks, FakeLLM(reply), password="secreta", **kwargs)


def ask_payload(service, **overrides):
    return {
        "question": "¿Vacaciones?",
        "ids": [17, 4],
        "fingerprint": service.fingerprint,
    } | overrides


def test_service_requires_a_password(chunks):
    with pytest.raises(ValueError):
        GenerationService(chunks, FakeLLM("x"), password="")


def test_service_checks_the_password(chunks):
    service = make_service(chunks)

    assert service.handle({"check": True}, "secreta", "c")[0] == 204
    assert service.handle({"check": True}, "otra", "c")[0] == 401
    assert service.handle(ask_payload(service), "", "c")[0] == 401


def test_service_answers_with_the_server_format(chunks):
    service = make_service(chunks)

    status, data = service.handle(ask_payload(service), "secreta", "c")

    assert status == 200
    assert data["mode"] == "generative"
    assert data["answer"] == "Respuesta [1]."
    assert data["sources"][0]["label"].startswith("manual-del-empleado.md")


def test_service_reports_abstention(chunks):
    service = make_service(chunks, reply=NO_ANSWER)

    status, data = service.handle(ask_payload(service), "secreta", "c")

    assert status == 200
    assert data["answered"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": ""},
        {"question": "x" * 3000},
        {"ids": []},
        {"ids": "17"},
        {"ids": [17, -1]},
        {"ids": [17, 10_000]},
        {"ids": [True]},
        {"ids": list(range(9))},
    ],
)
def test_service_rejects_invalid_requests(chunks, overrides):
    service = make_service(chunks)
    assert service.handle(ask_payload(service, **overrides), "secreta", "c")[0] == 400


def test_service_rejects_a_different_collection(chunks):
    service = make_service(chunks)
    status, _ = service.handle(ask_payload(service, fingerprint="otra"), "secreta", "c")
    assert status == 409


def test_service_limits_requests_per_client(chunks):
    service = make_service(chunks, rate_limit=2)

    assert service.handle({"check": True}, "secreta", "a")[0] == 204
    assert service.handle({"check": True}, "mala", "a")[0] == 401
    assert service.handle({"check": True}, "secreta", "a")[0] == 429
    assert service.handle({"check": True}, "secreta", "b")[0] == 204


def test_service_reports_provider_errors(chunks):
    class FailingLLM:
        def generate(self, system, prompt):
            raise LLMError("Proveedor no disponible")

    service = GenerationService(chunks, FailingLLM(), password="secreta")
    status, data = service.handle(ask_payload(service), "secreta", "c")
    assert (status, data["error"]) == (502, "Proveedor no disponible")


# --- Función de Vercel (api/answer.py) --------------------------------------------


@pytest.fixture
def function_url(chunks):
    spec = importlib.util.spec_from_file_location("answer_function", ROOT / "api" / "answer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._service = make_service(chunks)
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/answer", module._service
    server.shutdown()


def post(url, payload, password="secreta"):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Fundamenta-Password": password},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def test_function_answers_over_http(function_url):
    url, service = function_url

    status, headers, body = post(url, ask_payload(service))

    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body)["answer"] == "Respuesta [1]."


def test_function_handles_preflight_and_errors(function_url):
    url, service = function_url

    preflight = urllib.request.Request(url, method="OPTIONS")
    with urllib.request.urlopen(preflight) as response:
        assert response.status == 204
        assert "X-Fundamenta-Password" in response.headers["Access-Control-Allow-Headers"]

    assert post(url, ask_payload(service), password="mala")[0] == 401
    assert post(url, [1, 2])[0] == 400
