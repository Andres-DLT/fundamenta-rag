"""Función sin servidor (Vercel) que redacta las respuestas de la versión web.

Recibe la pregunta y los números de los fragmentos que el navegador ha recuperado, y
responde con el mismo formato que `POST /api/ask` del servidor. Toda la lógica está en
`fundamenta.web.GenerationService`; este archivo solo la adapta a HTTP.

Variables de entorno: FUNDAMENTA_PASSWORD (obligatoria), la clave del proveedor
(ANTHROPIC_API_KEY o GEMINI_API_KEY, entre otras), FUNDAMENTA_LLM (por defecto, el
proveedor de la clave configurada), FUNDAMENTA_RATE_LIMIT y FUNDAMENTA_ALLOWED_ORIGIN
(origen de la página web, por defecto cualquiera).
Ver docs/despliegue.md.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamenta.web import GenerationService  # noqa: E402

_service: GenerationService | None = None


def get_service() -> GenerationService:
    """El servicio se crea una vez por instancia y se reutiliza entre peticiones."""
    global _service
    if _service is None:
        _service = GenerationService.from_environment(ROOT)
    return _service


class handler(BaseHTTPRequestHandler):  # nombre que exige Vercel
    def _send(self, status: int, payload: dict | None = None) -> None:
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header(
            "Access-Control-Allow-Origin", os.environ.get("FUNDAMENTA_ALLOWED_ORIGIN", "*")
        )
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Fundamenta-Password")
        self.send_header("Access-Control-Max-Age", "86400")
        if payload is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204)

    def do_POST(self) -> None:  # noqa: N802
        try:
            service = get_service()
        except Exception as error:  # configuración incompleta: se informa en lugar de caer
            self._send(500, {"error": f"Servicio de generación mal configurado: {error}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            self._send(400, {"error": "JSON no válido"})
            return
        forwarded = self.headers.get("X-Forwarded-For", "")
        client = forwarded.split(",")[0].strip() or self.client_address[0]
        password = self.headers.get("X-Fundamenta-Password", "")
        self._send(*service.handle(payload, password, client))
