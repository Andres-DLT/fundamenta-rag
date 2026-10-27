# Capítulo 01: Cargar documentos

> **En este capítulo** escribirás la primera pieza del sistema: un módulo que lee
> archivos `.txt`, `.md` y `.pdf` y los convierte en objetos `Document` con su texto
> y sus metadatos.

📓 Notebook: [`notebook.ipynb`](notebook.ipynb) · 🧩 Código: [`rag/loaders.py`](../../rag/loaders.py)
· ✅ Tests: [`tests/test_loaders.py`](../../tests/test_loaders.py)

---

## Por qué importa este paso

En RAG se cumple la regla de *"basura entra, basura sale"*. Si el texto que extraemos
está mal (palabras pegadas, tablas rotas, encabezados repetidos en cada página),
la búsqueda encontrará peores fragmentos y el LLM dará peores respuestas, por muy
bueno que sea el modelo.

En las empresas, los datos vienen en muchos formatos: PDFs, Word, páginas de una wiki,
tickets, correos... En este curso empezamos con los tres más comunes:

| Formato | Ejemplo en NovaTech | Dificultad |
|---------|---------------------|------------|
| `.txt` | `faq-soporte-it.txt` | Fácil: el texto ya está "limpio". |
| `.md` | `manual-del-empleado.md` | Fácil, y además tiene estructura (títulos, tablas). |
| `.pdf` | `beneficios-2026.pdf` | Difícil: un PDF guarda *dónde se dibuja* cada letra, no párrafos. |

## La pieza central: `Document`

Todo lo que cargamos se convierte en un mismo tipo de objeto:

```python
@dataclass
class Document:
    text: str
    metadata: dict[str, Any]
```

- `text` es el contenido que usaremos para buscar y para construir el prompt.
- `metadata` dice **de dónde viene** ese texto, por ejemplo:

```python
{"source": "beneficios-2026.pdf", "type": "pdf", "page": 2}
```

Los metadatos parecen un detalle, pero son los que permitirán al sistema terminar
sus respuestas con *"Fuente: beneficios-2026.pdf, página 2"*. En los sistemas reales
también se usan para filtrar (por ejemplo, que cada persona solo pueda buscar en los
documentos a los que tiene acceso).

## Los loaders

El módulo tiene una función por formato y una función que elige la adecuada según la
extensión del archivo:

```python
LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path):
    loader = LOADERS.get(Path(path).suffix.lower())
    ...
    return loader(path)
```

Decisiones de diseño que vale la pena notar:

- **Siempre devolvemos una lista de `Document`**, aunque el archivo sea uno solo.
  Así los PDFs pueden devolver un documento por página sin cambiar la interfaz.
- **Un documento por página en los PDFs.** Hace posible citar la página exacta.
- **Markdown guarda su título** (`# ...`) en los metadatos. Será útil para dar
  contexto a cada chunk en el capítulo 02.
- **`normalize()` hace una limpieza mínima**: unifica los saltos de línea de Windows
  (`\r\n`), quita espacios sobrantes y colapsa las líneas en blanco repetidas.
  No hacemos más a propósito: limpiar de más también puede romper información.
- **`load_directory()` ignora los formatos desconocidos** en vez de fallar, y recorre
  los archivos en orden alfabético para que el resultado sea siempre el mismo.

## Lo que puede salir mal (y cómo se resuelve en el mundo real)

- **PDFs escaneados.** Son imágenes, no tienen texto. `load_pdf` las descarta; en
  producción se pasa antes un OCR (por ejemplo, Tesseract).
- **Tablas en PDFs.** Suelen extraerse como líneas sueltas sin estructura. Existen
  herramientas especializadas para extraerlas.
- **Encabezados y pies de página** repetidos en cada página, que meten ruido en la
  búsqueda.
- **Codificación del texto.** Aquí asumimos UTF-8; los archivos antiguos a veces
  vienen en Latin-1 o Windows-1252 y las tildes salen rotas (`vacaciÃ³n`).

## Pruébalo

```bash
pytest tests/test_loaders.py
```

En el [notebook](notebook.ipynb) cargarás toda la carpeta `data/novatech/` y verás
cuántos documentos y cuánto texto hay.

## Ejercicios

1. **Loader de HTML.** Añade `load_html()` usando `html.parser` de la librería
   estándar para quedarte solo con el texto visible. Regístralo en `LOADERS` para
   `.html`.
2. **Loader de CSV.** Escribe `load_csv()` que devuelva un `Document` por fila, con
   el formato `columna: valor` en el texto y el número de fila en los metadatos.
3. **Fecha de modificación.** Añade a los metadatos la fecha de última modificación
   del archivo (`path.stat().st_mtime`). ¿Para qué podría servir al responder?
4. **Detectar PDFs escaneados.** Haz que `load_pdf` avise con un `warning` cuando una
   página no tenga texto.

Recuerda añadir un test por cada cambio.

---

⬅️ **Anterior:** [Capítulo 00: ¿Qué es RAG?](../00-que-es-rag/README.md)
· ➡️ **Siguiente:** Capítulo 02: Chunking *(próximamente)*
