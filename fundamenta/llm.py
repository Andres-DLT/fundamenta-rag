"""Módulo 07: proveedores de modelos de lenguaje (LLM).

Todos los proveedores implementan la misma interfaz mínima, `generate(system, prompt)`,
de modo que el resto del sistema no depende de ningún proveedor concreto:

- `ClaudeLLM`: Claude, de Anthropic, mediante su SDK oficial (extra `[claude]`).
- `OllamaLLM`: cualquier modelo abierto ejecutado en local con Ollama. Los datos no
  salen de la máquina.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class LLMError(RuntimeError):
    """Error al obtener una respuesta del proveedor."""


class ClaudeLLM:
    """Claude a través de la API de Anthropic.

    Las credenciales se leen del entorno (`ANTHROPIC_API_KEY` o un perfil de
    `ant auth login`). Se activa el mecanismo de respaldo del servidor
    (`fallbacks="default"`): si el modelo declina una petición, la API la reintenta
    con otro modelo dentro de la misma llamada.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
    ) -> None:
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                'ClaudeLLM requiere el extra [claude]: pip install -e ".[claude]"'
            ) from error
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self._anthropic.APIError as error:
            raise LLMError(f"Error de la API de Anthropic: {error}") from error

        if response.stop_reason == "refusal":
            raise LLMError("El modelo declinó responder a la petición")
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaLLM:
    """Modelo abierto servido en local por Ollama (https://ollama.com).

    Requiere tener Ollama en ejecución y el modelo descargado, por ejemplo:
    `ollama pull qwen2.5:7b`.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise LLMError(f"No se pudo contactar con Ollama en {self.host}: {error}") from error
        return body["message"]["content"].strip()


def get_llm(name: str | None) -> LLM | None:
    """Crea un LLM a partir de un nombre.

    - `None` o `"none"`: sin LLM (modo extractivo).
    - `"claude"` o `"claude:<modelo>"`.
    - `"ollama"` o `"ollama:<modelo>"`, por ejemplo `"ollama:llama3.1:8b"`.
    """
    if not name or name == "none":
        return None
    provider, _, model = name.partition(":")
    if provider == "claude":
        return ClaudeLLM(model=model) if model else ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM(model=model) if model else OllamaLLM()
    raise ValueError(f"LLM desconocido: {name!r} (opciones: none, claude, ollama)")
