"""Función de generación de la versión web, como aplicación WSGI (punto de entrada en Vercel).

Recibe en `POST /api/answer` la pregunta y los números de los fragmentos que el navegador
ha recuperado, y responde con el mismo formato que `POST /api/ask` del servidor. Toda la
lógica está en `fundamenta.web.GenerationService`; este archivo solo la adapta a HTTP.

Variables de entorno: FUNDAMENTA_PASSWORD (obligatoria), la clave del proveedor
(ANTHROPIC_API_KEY o GEMINI_API_KEY, entre otras), FUNDAMENTA_LLM (por defecto, el
proveedor de la clave configurada), FUNDAMENTA_RATE_LIMIT y FUNDAMENTA_ALLOWED_ORIGIN
(origen de la página web, por defecto cualquiera). Ver docs/despliegue.md.

En local puede probarse con la librería estándar:

    python -c "from wsgiref.simple_server import make_server; from app import app; \
make_server('127.0.0.1', 8000, app).serve_forever()"
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fundamenta.web import GenerationService  # noqa: E402

ROUTE = "/api/answer"

_service: GenerationService | None = None


def get_service() -> GenerationService:
    """El servicio se crea una vez por instancia y se reutiliza entre peticiones."""
    global _service
    if _service is None:
        _service = GenerationService.from_environment(ROOT)
    return _service


def dispatch(environ: dict) -> tuple[int, dict | None]:
    if environ.get("PATH_INFO", "").rstrip("/") != ROUTE:
        return 404, {"error": "Recurso no encontrado"}
    method = environ.get("REQUEST_METHOD", "GET")
    if method == "OPTIONS":
        return 204, None
    if method != "POST":
        return 405, {"error": "Utilice POST"}
    try:
        service = get_service()
    except Exception as error:  # configuración incompleta: se informa en lugar de caer
        return 500, {"error": f"Servicio de generación mal configurado: {error}"}
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        payload = json.loads(environ["wsgi.input"].read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError
    except ValueError:
        return 400, {"error": "JSON no válido"}
    forwarded = environ.get("HTTP_X_FORWARDED_FOR", "")
    client = forwarded.split(",")[0].strip() or environ.get("REMOTE_ADDR", "")
    password = environ.get("HTTP_X_FUNDAMENTA_PASSWORD", "")
    return service.handle(payload, password, client)


def app(environ: dict, start_response: Callable) -> Iterable[bytes]:
    status, payload = dispatch(environ)
    body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = [
        ("Access-Control-Allow-Origin", os.environ.get("FUNDAMENTA_ALLOWED_ORIGIN", "*")),
        ("Access-Control-Allow-Methods", "POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, X-Fundamenta-Password"),
        ("Access-Control-Max-Age", "86400"),
        ("Content-Length", str(len(body))),
    ]
    if payload is not None:
        headers.append(("Content-Type", "application/json; charset=utf-8"))
    start_response(f"{status} {HTTPStatus(status).phrase}", headers)
    return [body]
