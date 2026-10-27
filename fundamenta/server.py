"""Módulo 08: servidor web.

Expone el pipeline mediante una API HTTP mínima y una interfaz web, usando solo la
librería estándar de Python:

- `GET /`: interfaz de consulta.
- `GET /api/health`: estado del sistema (sin autenticación, para comprobaciones de salud).
- `POST /api/ask` con `{"question": "..."}`: respuesta con sus fuentes.

Para exponerlo en internet incorpora dos protecciones opcionales:

- Contraseña de acceso (autenticación HTTP Basic), para que solo accedan las personas
  invitadas a la demostración.
- Límite de preguntas por minuto y cliente, para acotar el consumo del proveedor del LLM.

Un despliegue corporativo debe añadir autenticación con el proveedor de identidad de la
organización (ver módulo 11).
"""

from __future__ import annotations

import base64
import hmac
import json
import threading
import time
from collections import deque
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


class RateLimiter:
    """Ventana deslizante: como máximo `limit` peticiones cada `window` segundos por cliente."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def check_password(header: str | None, password: str) -> bool:
    """Valida una cabecera `Authorization: Basic ...`. El nombre de usuario se ignora."""
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    _, _, given = decoded.partition(":")
    return hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8"))


def create_server(
    pipeline: RAGPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
    examples: list[str] | None = None,
    password: str | None = None,
    rate_limit: int = 30,
) -> ThreadingHTTPServer:
    """Crea (sin arrancar) el servidor HTTP. `port=0` elige un puerto libre.

    `password` activa la autenticación HTTP Basic en la interfaz y en la API.
    `rate_limit` es el número máximo de preguntas por minuto y cliente.
    """
    page = resources.files("fundamenta").joinpath("static/index.html").read_bytes()
    limiter = RateLimiter(rate_limit)
    health = {
        "status": "ok",
        "organization": pipeline.config.organization,
        "chunks": len(pipeline.chunks),
        "retrieval": pipeline.mode,
        "generation": type(pipeline.llm).__name__ if pipeline.llm else "extractive",
        "examples": examples or [],
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(
            self, status: HTTPStatus, payload: dict, headers: dict[str, str] | None = None
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", headers)

        def _client(self) -> str:
            # Detrás de un proxy (por ejemplo, la plataforma de despliegue), la IP real
            # del cliente llega en X-Forwarded-For.
            forwarded = self.headers.get("X-Forwarded-For", "")
            return forwarded.split(",")[0].strip() or self.client_address[0]

        def _authorized(self) -> bool:
            if password is None or check_password(self.headers.get("Authorization"), password):
                return True
            self._json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "Se requiere autenticación"},
                {"WWW-Authenticate": 'Basic realm="Fundamenta", charset="UTF-8"'},
            )
            return False

        def do_GET(self) -> None:  # noqa: N802 (nombre impuesto por http.server)
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._json(HTTPStatus.OK, health)
            elif not self._authorized():
                return
            elif path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Recurso no encontrado"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            if urlsplit(self.path).path != "/api/ask":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Recurso no encontrado"})
                return
            if not limiter.allow(self._client()):
                self._json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "Demasiadas preguntas seguidas. Espere un momento."},
                    {"Retry-After": str(int(limiter.window))},
                )
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
            print(f"[fundamenta] {self._client()} {format % args}", flush=True)

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    pipeline: RAGPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
    examples: list[str] | None = None,
    password: str | None = None,
    rate_limit: int = 30,
) -> None:
    """Arranca el servidor y atiende peticiones hasta que se interrumpe (Ctrl+C)."""
    server = create_server(pipeline, host, port, examples, password, rate_limit)
    protection = "con contraseña" if password else "sin contraseña"
    print(
        f"Fundamenta disponible en http://{host}:{server.server_address[1]} "
        f"({protection}, máximo {rate_limit} preguntas por minuto y cliente)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
