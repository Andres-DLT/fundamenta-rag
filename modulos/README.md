# Programa técnico

El programa recorre, en doce módulos, la construcción completa de un sistema RAG
(*Retrieval-Augmented Generation*). Cada módulo implementa una pieza del paquete
`fundamenta/` y se acompaña de un documento explicativo, un cuaderno ejecutable y
pruebas automatizadas.

| Módulo | Contenido | Código |
|---|---|---|
| [00 · Introducción](00-introduccion/README.md) | Qué es RAG, por qué existe y cuándo utilizarlo | — |
| [01 · Ingesta de documentos](01-ingesta/README.md) | Lectura de TXT, Markdown y PDF con metadatos | `loaders.py` |
| [02 · Segmentación](02-segmentacion/README.md) | División en fragmentos respetando la estructura | `chunking.py` |
| [03 · Búsqueda léxica](03-busqueda-lexica/README.md) | Tokenización para español y BM25 | `text.py`, `bm25.py` |
| [04 · Embeddings](04-embeddings/README.md) | Representación semántica: LSA y modelo multilingüe | `embeddings.py` |
| [05 · Almacén vectorial](05-almacen-vectorial/README.md) | Búsqueda por similitud, filtros y persistencia | `vector_store.py` |
| [06 · Recuperación híbrida](06-recuperacion-hibrida/README.md) | Fusión de rankings (RRF) y diversidad (MMR) | `retriever.py` |
| [07 · Generación con citas](07-generacion/README.md) | Prompt, proveedores de LLM y citas verificables | `generation.py`, `llm.py` |
| [08 · Pipeline y demostración](08-pipeline-y-demo/README.md) | API de alto nivel, CLI y aplicación web | `pipeline.py`, `cli.py`, `server.py` |
| [09 · Evaluación](09-evaluacion/README.md) | Métricas de recuperación y de respuesta | `evaluation.py` |
| [10 · Mejoras de calidad](10-mejoras/README.md) | Ajustes medidos y reescritura de consultas | `query.py` |
| [11 · Puesta en producción](11-produccion/README.md) | Escalado, seguridad, operación y costes | — |

## Cómo seguir el programa

1. Instale el proyecto siguiendo las instrucciones del [README principal](../README.md).
2. Lea el documento de cada módulo y ejecute su cuaderno (`jupyter lab`).
3. Revise el código correspondiente en `fundamenta/` y sus pruebas en `tests/`.
4. Resuelva los ejercicios propuestos al final de cada módulo.

Los módulos 04 a 06 y 08 a 10 utilizan el modelo de embeddings multilingüe, que
requiere el extra `[semantic]`.
