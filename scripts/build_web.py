"""Genera la versión web de Fundamenta: una página estática que busca en el navegador.

El resultado puede publicarse en cualquier alojamiento de páginas estáticas (GitHub
Pages, Netlify, un bucket de almacenamiento...). Ver docs/despliegue.md.

Requisitos:
    pip install -e ".[onnx]"

Uso:
    python scripts/build_web.py --out site
    python scripts/build_web.py --out site --generation-url https://<proyecto>.vercel.app/api/answer

Sin `--generation-url`, la página responde en modo extractivo. Con ella, envía la
pregunta y los fragmentos recuperados a la función de generación (api/answer.py), que
redacta la respuesta con Claude y exige la contraseña de acceso.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from fundamenta.cli import DEMO_EXAMPLES
from fundamenta.embeddings import OnnxEmbedder
from fundamenta.pipeline import PipelineConfig
from fundamenta.web import build_index, load_chunks, write_index

ROOT = Path(__file__).resolve().parents[1]
ENGINE_TAG = '<script src="engine.js"></script>\n<script>\n  const $ ='


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(ROOT / "data" / "novatech"))
    parser.add_argument("--organization", default="NovaTech")
    parser.add_argument("--out", default="site", help="carpeta de salida")
    parser.add_argument(
        "--generation-url",
        default=os.environ.get("FUNDAMENTA_GENERATION_URL", ""),
        help="dirección de la función de generación (opcional)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    config = PipelineConfig(organization=args.organization)
    chunks = load_chunks(args.data, config)
    examples = DEMO_EXAMPLES if Path(args.data).name == "novatech" else []
    index = build_index(chunks, OnnxEmbedder(), config, examples)
    index["generation"] = {"url": args.generation_url} if args.generation_url else None
    write_index(index, out / "index.json")

    page = (ROOT / "fundamenta" / "static" / "index.html").read_text(encoding="utf-8")
    if page.count("<script>\n  const $ =") != 1:
        raise SystemExit("No se encontró el script principal en index.html")
    page = page.replace("<script>\n  const $ =", ENGINE_TAG)
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copy(ROOT / "web" / "engine.js", out / "engine.js")
    (out / ".nojekyll").touch()  # GitHub Pages: publicar los archivos tal cual

    mode = f"generación en {args.generation_url}" if args.generation_url else "modo extractivo"
    print(f"Web generada en {out}: {len(chunks)} fragmentos, {mode}")


if __name__ == "__main__":
    main()
