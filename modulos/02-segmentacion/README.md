# Módulo 02 · Segmentación (chunking)

> División de los documentos en fragmentos que puedan recuperarse de forma precisa,
> con tres estrategias de complejidad creciente.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/chunking.py`](../../fundamenta/chunking.py)
· [pruebas](../../tests/test_chunking.py)

---

## Objetivos

- Entender por qué los documentos deben dividirse antes de indexarse.
- Comparar la segmentación por tamaño fijo, recursiva y estructural.
- Conservar el contexto (documento y sección) de cada fragmento.

## Motivación

Para responder "¿cuántos días de vacaciones tengo?" basta un párrafo del manual, no el
documento completo. Dividir los documentos aporta tres ventajas:

1. **Búsquedas más precisas.** Un fragmento trata un único tema; un documento largo
   mezcla varios y su representación semántica se diluye.
2. **Prompts más breves**, con menor coste y latencia.
3. **Escalabilidad**: solo se envían al modelo los fragmentos relevantes.

## El compromiso del tamaño

| | Fragmentos pequeños | Fragmentos grandes |
|---|---|---|
| Precisión de la búsqueda | Alta | Baja |
| Contexto de cada fragmento | Puede quedar incompleto | Ideas completas |
| Tamaño del prompt | Reducido | Elevado |

Se utilizan **500 caracteres con 50 de solapamiento** (unos 125 tokens). El módulo 10
mide el efecto de otros valores.

## Estrategias

### 1. Tamaño fijo (`chunk_fixed`)

Corta el texto cada `chunk_size` caracteres. El solapamiento repite el final de cada
fragmento al inicio del siguiente para no perder lo que cae en un corte. Es simple y
predecible, pero parte palabras y frases.

### 2. Recursiva por separadores (`chunk_recursive`)

Divide por el separador de mayor nivel posible —párrafo, línea, frase, palabra— hasta
que cada pieza cabe en el tamaño indicado, y después agrupa las piezas consecutivas.
El solapamiento se aplica con piezas completas, sin cortar palabras. Es el mismo
principio que aplica el `RecursiveCharacterTextSplitter` de LangChain.

### 3. Estructural (`split_markdown_sections`)

Aprovecha la estructura que los autores ya han dado al documento: cada sección de
Markdown se trata por separado y se registra su ruta de títulos, por ejemplo
`Manual del empleado de NovaTech > Vacaciones`.

### Integración (`chunk_documents`)

Divide los documentos Markdown por secciones y aplica la estrategia recursiva a cada
sección, a los archivos de texto y a cada página de PDF. Cada fragmento hereda los
metadatos de su documento y añade su posición (`chunk`) y su sección (`section`).
Con la configuración por defecto, el caso de estudio pasa de 6 documentos a
41 fragmentos.

## Pérdida de contexto

El fragmento que contiene la tabla de permisos retribuidos no incluye la palabra
"permiso": esa información está en el título de su sección. Para que la búsqueda lo
localice, la función `contextualize` antepone la sección al texto que se indexa
(módulo 03). Por el mismo motivo se descartan los fragmentos que solo contienen un
título.

## Limitaciones

- Las tablas que no caben en un fragmento se separan de su cabecera.
- Un carácter `#` dentro de un bloque de código se interpreta como título.
- No se detectan cambios de tema dentro de una sección (segmentación semántica).

## Ejercicios propuestos

1. Implementar `chunk_sentences(text, sentences_per_chunk=3, overlap=1)`.
2. Ignorar los títulos dentro de bloques de código en `split_markdown_sections`.
3. Repetir la cabecera de una tabla en cada fragmento que contenga parte de ella.
4. Medir cuántos fragmentos de `chunk_fixed` terminan a mitad de palabra.

---

[← Módulo 01 · Ingesta](../01-ingesta/README.md) · [Índice](../README.md)
· [Módulo 03 · Búsqueda léxica →](../03-busqueda-lexica/README.md)
