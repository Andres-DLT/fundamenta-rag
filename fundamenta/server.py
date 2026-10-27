"""Módulo 08: servidor web de demostración.

Expone el pipeline mediante una API HTTP mínima y una interfaz web, usando solo la
librería estándar de Python:

- `GET /`: interfaz de consulta.
- `GET /api/health`: estado del sistema.
- `POST /api/ask` con `{"question": "..."}`: respuesta con sus fuentes.

Pensado para demostraciones y entornos internos. En producción se recomienda un
servidor de aplicaciones con autenticación delante (ver módulo 11).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlsplit

from fundamenta.generation import Answer
from fundamenta.pipeline import RAGPipeline

MAX_QUESTION_LENGTH = 2000


def answer_to_dict(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "answer": answer.text,
        "answered": answer.answered,
        "mode": answer.mode,
        "sources": [{**asdict(source), "label": source.label} for source in answer.sources],
    }


def create_server(
    pipeline: RAGPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
    examples: list[str] | None = None,
) -> ThreadingHTTPServer:
    """Crea (sin arrancar) el servidor HTTP. `port=0` elige un puerto libre."""
    page = resources.files("fundamenta").joinpath("static/index.html").read_bytes()
    health = {
        "status": "ok",
        "organization": pipeline.config.organization,
        "chunks": len(pipeline.chunks),
        "retrieval": pipeline.mode,
        "generation": type(pipeline.llm).__name__ if pipeline.llm else "extractive",
        "examples": examples or [],
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (nombre impuesto por http.server)
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(HTTPStatus.OK, health)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Recurso no encontrado"})

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/api/ask":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Recurso no encontrado"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                question = str(payload.get("question", "")).strip()
            except (ValueError, json.JSONDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "JSON no válido"})
                return
            if not question or len(question) > MAX_QUESTION_LENGTH:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"La pregunta debe tener entre 1 y {MAX_QUESTION_LENGTH} caracteres"},
                )
                return
            try:
                answer = pipeline.ask(question)
            except Exception as error:  # el detalle se registra, no se expone al cliente
                self.log_error("Error al responder: %r", error)
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "No se pudo generar la respuesta. Consulte el registro."},
                )
                return
            self._json(HTTPStatus.OK, answer_to_dict(answer))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            print(f"[fundamenta] {self.address_string()} {format % args}")

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    pipeline: RAGPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
    examples: list[str] | None = None,
) -> None:
    """Arranca el servidor y atiende peticiones hasta que se interrumpe (Ctrl+C)."""
    server = create_server(pipeline, host, port, examples)
    print(f"Fundamenta disponible en http://{host}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
