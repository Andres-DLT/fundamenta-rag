"""Despliega Fundamenta como un Space de Hugging Face (contenedor Docker).

Requisitos:
    pip install -e ".[deploy]"
    hf auth login            # una sola vez, con un token de escritura

Uso:
    python scripts/deploy_huggingface.py --space <usuario>/fundamenta
    python scripts/deploy_huggingface.py --space <usuario>/fundamenta --llm claude

Las credenciales (ANTHROPIC_API_KEY, FUNDAMENTA_PASSWORD) no se pasan a este script:
se configuran como secretos en la página del Space (Settings > Variables and secrets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

# Archivos necesarios para construir la imagen (ver Dockerfile)
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "LICENSE", "fundamenta/**", "data/**"]

SPACE_README = """---
title: Fundamenta
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Respuestas fundamentadas en su documentación, con citas
---

# Fundamenta

Motor RAG con respuestas fundamentadas en la documentación de la organización y citas
verificables. Esta demostración utiliza el caso de estudio de NovaTech, una empresa
ficticia.

Autor: Andrés de la Torre Flores · [Repositorio](https://github.com/Andres-DLT/fundamenta-rag)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", required=True, help="identificador del Space: usuario/nombre")
    parser.add_argument(
        "--llm", default="none", help="FUNDAMENTA_LLM del Space: none, claude u ollama:<modelo>"
    )
    parser.add_argument("--private", action="store_true", help="crear el Space como privado")
    args = parser.parse_args()

    api = HfApi()
    print(f"Usuario de Hugging Face: {api.whoami()['name']}")

    api.create_repo(
        args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=FILES,
        ignore_patterns=["**/__pycache__/**"],
        commit_message="Despliegue de Fundamenta",
    )
    api.upload_file(
        repo_id=args.space,
        repo_type="space",
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Descripción del Space",
    )
    api.add_space_variable(args.space, "FUNDAMENTA_LLM", args.llm)

    owner, name = args.space.split("/")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    print(f"Aplicación: https://{owner}-{name}.hf.space".lower().replace("_", "-"))
    print("La construcción de la imagen tarda unos minutos la primera vez.")


if __name__ == "__main__":
    main()
