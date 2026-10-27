# Módulo 01 · Ingesta de documentos

> Conversión de archivos TXT, Markdown y PDF en objetos `Document` que combinan el
> texto con los metadatos necesarios para citar y filtrar.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/loaders.py`](../../fundamenta/loaders.py)
· [pruebas](../../tests/test_loaders.py)

---

## Objetivos

- Definir una representación común para cualquier documento.
- Extraer texto de formatos con distinta dificultad.
- Conservar el origen de cada texto para poder citarlo.

## Relevancia de la ingesta

La calidad de un sistema RAG está limitada por la calidad del texto que se extrae.
Palabras unidas, tablas desordenadas o encabezados repetidos en cada página degradan
la búsqueda y, por tanto, las respuestas, con independencia del modelo utilizado.

| Formato | Ejemplo en el caso de estudio | Dificultad |
|---|---|---|
| `.txt` | `faq-soporte-it.txt` | Baja: el texto ya está disponible. |
| `.md` | `manual-del-empleado.md` | Baja, y aporta estructura (títulos, tablas). |
| `.pdf` | `beneficios-2026.pdf` | Alta: el PDF describe la posición de cada carácter, no párrafos. |

## Implementación

### La clase `Document`

```python
@dataclass
class Document:
    text: str
    metadata: dict[str, Any]
```

Los metadatos indican el origen del texto, por ejemplo
`{"source": "beneficios-2026.pdf", "type": "pdf", "page": 2}`. Permiten que las
respuestas citen "beneficios-2026.pdf, página 2" y que la búsqueda se restrinja a los
documentos que cada usuario está autorizado a consultar.

### Lectores por formato

```python
LOADERS = {".txt": load_text, ".md": load_markdown, ".pdf": load_pdf}
```

`load_document` selecciona el lector según la extensión y `load_directory` procesa una
carpeta completa en orden alfabético, ignorando los formatos no soportados.

## Decisiones de diseño

- **Todos los lectores devuelven una lista de documentos.** Un PDF devuelve un
  documento por página sin alterar la interfaz.
- **Un documento por página en PDF**, para citar la página exacta.
- **El título del Markdown se guarda en los metadatos** y se aprovecha en la
  segmentación (módulo 02).
- **Normalización mínima** (`normalize`): saltos de línea uniformes, sin espacios
  sobrantes ni líneas en blanco repetidas. Una limpieza más agresiva puede eliminar
  información útil.

## Limitaciones y tratamiento en entornos reales

| Situación | Tratamiento habitual |
|---|---|
| PDF escaneados (imágenes sin texto) | Reconocimiento óptico de caracteres (OCR) previo |
| Tablas complejas en PDF | Extractores especializados de tablas |
| Encabezados y pies repetidos | Detección y eliminación por frecuencia entre páginas |
| Codificaciones antiguas (Latin-1, Windows-1252) | Detección de codificación al leer |
| Word, HTML, wikis, tickets | Lectores adicionales con la misma interfaz |

## Ejercicios propuestos

1. Añadir un lector de HTML con `html.parser` de la librería estándar.
2. Añadir un lector de CSV que genere un documento por fila con el formato `columna: valor`.
3. Incorporar la fecha de modificación del archivo a los metadatos.
4. Emitir un aviso cuando una página de un PDF no contenga texto.

---

[← Módulo 00 · Introducción](../00-introduccion/README.md) · [Índice](../README.md)
· [Módulo 02 · Segmentación →](../02-segmentacion/README.md)
