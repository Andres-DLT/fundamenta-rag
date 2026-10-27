# Módulo 08 · Pipeline, línea de comandos y demostración web

> Integración de todas las piezas en una interfaz de alto nivel, una herramienta de
> línea de comandos y una aplicación web de demostración.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/pipeline.py`](../../fundamenta/pipeline.py)
· [`fundamenta/cli.py`](../../fundamenta/cli.py) · [`fundamenta/server.py`](../../fundamenta/server.py)
· [pruebas](../../tests/test_application.py)

---

## API de Python

```python
from fundamenta import ClaudeLLM, PipelineConfig, RAGPipeline, SentenceTransformerEmbedder

pipeline = RAGPipeline.from_directory(
    "data/novatech",
    embedder=SentenceTransformerEmbedder(),
    llm=ClaudeLLM(),
    config=PipelineConfig(organization="NovaTech", k=4),
)

answer = pipeline.ask("¿Cuántos días de vacaciones tengo?")
print(answer.text)
for source in answer.sources:
    print(source.number, source.label)
```

| Parámetro de `PipelineConfig` | Función | Por defecto |
|---|---|---|
| `chunk_size`, `overlap` | Segmentación (módulo 02) | 500, 50 |
| `k` | Fragmentos que se envían al modelo | 4 |
| `mode` | `lexical`, `semantic` o `hybrid` | `hybrid` |
| `contextual` | Indexar la sección junto al texto | `True` |
| `lexical_weight` | Peso de BM25 en la fusión | 0,5 |
| `rewrite_queries` | Reformular la pregunta con el LLM (módulo 10) | `False` |
| `organization` | Nombre que se usa en las instrucciones del asistente | "la organización" |

Sin embedder, el pipeline utiliza búsqueda léxica; sin LLM, responde en modo
extractivo. Así, el sistema funciona en cualquier entorno y mejora a medida que se
habilitan componentes.

## Línea de comandos

```bash
fundamenta ask "¿Cuántos días de vacaciones tengo?" --embedder semantic --llm claude
fundamenta search "dietas en viajes" --k 3
fundamenta serve --embedder semantic --llm claude --port 8000
fundamenta eval --embedder semantic
```

| Opción | Valores |
|---|---|
| `--data` | Carpeta de documentos (por defecto, `data/novatech`) |
| `--embedder` | `lsa` (sin descargas), `semantic` (modelo E5) o `none` |
| `--llm` | `none`, `claude`, `claude:<modelo>`, `ollama` u `ollama:<modelo>` |
| `--mode`, `--k`, `--chunk-size`, `--overlap`, `--organization` | Parámetros del pipeline |

## Aplicación web de demostración

`fundamenta serve` arranca un servidor HTTP basado únicamente en la librería estándar
de Python:

| Ruta | Función |
|---|---|
| `GET /` | Interfaz de consulta |
| `GET /api/health` | Estado: organización, número de fragmentos, modo de búsqueda y generación |
| `POST /api/ask` | `{"question": "..."}` → respuesta, modo y fuentes con su texto |

La interfaz muestra la respuesta con las citas como enlaces a sus fuentes, el
fragmento original de cada fuente y su puntuación de relevancia. Se adapta al modo
claro u oscuro del sistema y a pantallas móviles.

El servidor valida la longitud de las preguntas y no expone detalles internos en los
errores. Para publicarlo en internet incorpora dos protecciones:

- `--password` (o `FUNDAMENTA_PASSWORD`): contraseña de acceso mediante autenticación
  HTTP Basic, para que solo accedan las personas invitadas.
- `--rate-limit` (o `FUNDAMENTA_RATE_LIMIT`): número máximo de preguntas por minuto y
  cliente, que acota el consumo del proveedor del modelo.

La [guía de despliegue](../../docs/despliegue.md) explica cómo publicarlo con Docker o
Hugging Face Spaces, y el módulo 11 describe los requisitos adicionales de un despliegue
corporativo (autenticación con el proveedor de identidad, control de acceso por
documento, registro de consultas).

## Ejercicios propuestos

1. Añadir a la API un parámetro `source` para filtrar por documento.
2. Registrar en un archivo cada pregunta, las fuentes recuperadas y el tiempo de respuesta.
3. Persistir el índice con `VectorStore.save` y cargarlo al arrancar el servidor.

---

[← Módulo 07 · Generación con citas](../07-generacion/README.md) · [Índice](../README.md)
· [Módulo 09 · Evaluación →](../09-evaluacion/README.md)
