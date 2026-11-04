"""Módulo 07: proveedores de modelos de lenguaje (LLM).

Todos los proveedores implementan la misma interfaz mínima, `generate(system, prompt)`,
de modo que el resto del sistema no depende de ningún proveedor concreto:

- `ClaudeLLM`: Claude, de Anthropic, mediante su SDK oficial (extra `[claude]`).
- `OllamaLLM`: cualquier modelo abierto ejecutado en local con Ollama. Los datos no
  salen de la máquina.
- `OpenAICompatibleLLM`: cualquier proveedor con la API de Chat Completions de OpenAI,
  como Google Gemini, OpenAI o pasarelas como OpenRouter y Groq.
"""

from __future__ import annotations

import json
import os
import time
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


class TransientLLMError(LLMError):
    """Error pasajero del proveedor (saturación, límite de uso, fallo temporal)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# Códigos que indican que conviene reintentar o pasar a otro modelo. El 404 suele
# significar que el modelo se ha retirado: se pasa directamente al siguiente.
TRANSIENT_STATUS = {404, 408, 429, 500, 502, 503, 504}


def _error_message(raw: str) -> str:
    """Mensaje legible de una respuesta de error, venga como objeto o como lista."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw[:200]
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    message = error.get("message", error) if isinstance(error, dict) else error
    return str(message)[:200]


class OpenAICompatibleLLM:
    """Cualquier proveedor que ofrezca la API de Chat Completions de OpenAI.

    Es el formato más extendido: lo admiten OpenAI, Google Gemini y la mayoría de
    pasarelas de modelos. Solo usa la librería estándar.

    Si el modelo está saturado o no disponible, se reintenta una vez y después se prueba
    con cada uno de los `fallback_models`, sin superar `deadline` segundos en total.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        label: str = "LLM",
        fallback_models: tuple[str, ...] = (),
        timeout: float = 30.0,
        deadline: float = 50.0,
        retry_delay: float = 1.5,
    ) -> None:
        if not api_key:
            raise ValueError(f"Falta la clave de API de {label}")
        self.model = model
        self.fallback_models = tuple(fallback_models)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.label = label
        self.timeout = timeout
        self.deadline = deadline
        self.retry_delay = retry_delay

    def generate(self, system: str, prompt: str) -> str:
        deadline = time.monotonic() + self.deadline
        failures = []
        for index, model in enumerate((self.model, *self.fallback_models)):
            tries = 2 if index == 0 else 1
            for attempt in range(tries):
                remaining = deadline - time.monotonic()
                if remaining < 2:
                    break
                try:
                    return self._complete(model, system, prompt, min(self.timeout, remaining))
                except TransientLLMError as error:
                    failures.append(f"{model}: {error}")
                    if error.status == 404:
                        break
                    if attempt + 1 < tries:
                        time.sleep(min(self.retry_delay, max(0.0, remaining - 2)))
        detail = "; ".join(failures) or "sin tiempo para responder"
        raise LLMError(f"{self.label} no está disponible en este momento ({detail})")

    def _complete(self, model: str, system: str, prompt: str, timeout: float) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            message = _error_message(error.read().decode("utf-8", errors="replace"))
            text = f"Error {error.code} de {self.label}: {message}"
            if error.code in TRANSIENT_STATUS:
                raise TransientLLMError(text, error.code) from error
            raise LLMError(text) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise TransientLLMError(f"No se pudo contactar con {self.label}: {error}") from error

        choice = body["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("El modelo declinó responder a la petición")
        return (choice["message"].get("content") or "").strip()


# Proveedores con API compatible con OpenAI: dirección, variable de la clave y modelos
# por defecto, el principal seguido de los de respaldo (vacío obliga a indicarlos, por
# ejemplo "openai:<modelo>"). Los de Gemini están disponibles en su plan gratuito.
OPENAI_COMPATIBLE = {
    "gemini": (
        "Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"),
    ),
    "openai": ("OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY", ()),
}


def get_llm(name: str | None) -> LLM | None:
    """Crea un LLM a partir de un nombre.

    - `None` o `"none"`: sin LLM (modo extractivo).
    - `"claude"` o `"claude:<modelo>"`.
    - `"ollama"` o `"ollama:<modelo>"`, por ejemplo `"ollama:llama3.1:8b"`.
    - `"gemini"` o `"gemini:<modelo>"`, con la clave en `GEMINI_API_KEY`.
    - Varios modelos separados por comas (`"gemini:<modelo>,<respaldo>"`) se prueban en
      orden cuando el anterior está saturado.
    - `"openai:<modelo>"`, con la clave en `OPENAI_API_KEY`.
    - `"compatible:<modelo>"`: cualquier otro proveedor compatible con OpenAI, con la
      dirección en `FUNDAMENTA_LLM_BASE_URL` y la clave en `FUNDAMENTA_LLM_API_KEY`.
    """
    if not name or name == "none":
        return None
    provider, _, model = name.partition(":")
    if provider == "claude":
        return ClaudeLLM(model=model) if model else ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM(model=model) if model else OllamaLLM()
    if provider in OPENAI_COMPATIBLE:
        label, base_url, key_variable, default_models = OPENAI_COMPATIBLE[provider]
    elif provider == "compatible":
        label, base_url = "el proveedor", os.environ.get("FUNDAMENTA_LLM_BASE_URL", "")
        key_variable, default_models = "FUNDAMENTA_LLM_API_KEY", ()
        if not base_url:
            raise ValueError("Falta FUNDAMENTA_LLM_BASE_URL para el proveedor compatible")
    else:
        raise ValueError(
            f"LLM desconocido: {name!r} "
            "(opciones: none, claude, ollama, gemini, openai, compatible)"
        )
    models = [m.strip() for m in model.split(",") if m.strip()] or list(default_models)
    if not models:
        raise ValueError(f"Indique el modelo, por ejemplo {provider}:<modelo>")
    return OpenAICompatibleLLM(
        model=models[0],
        fallback_models=tuple(models[1:]),
        base_url=base_url,
        api_key=os.environ.get(key_variable, ""),
        label=label,
    )


def default_llm_name() -> str:
    """Proveedor que corresponde a la clave configurada en el entorno."""
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return "gemini"
    return "claude"
