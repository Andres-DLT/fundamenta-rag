# Módulo 03 · Búsqueda léxica con BM25

> Implementación del algoritmo de ranking de referencia de los motores de búsqueda,
> con una tokenización adaptada al español.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/text.py`](../../fundamenta/text.py)
· [`fundamenta/bm25.py`](../../fundamenta/bm25.py) · [pruebas](../../tests/test_search.py)

---

## Objetivos

- Normalizar el texto para que las variantes de una palabra coincidan.
- Comprender los componentes de BM25: frecuencia, rareza y longitud.
- Identificar en qué consultas falla la búsqueda léxica.

## Tokenización para búsqueda

`tokenize` transforma un texto en una lista de términos:

1. Minúsculas y eliminación de tildes ("Información" → "informacion").
2. Eliminación de palabras vacías ("de", "la", "que", "cuántos"…).
3. *Stemming* ligero (`light_stem`): elimina el plural y la vocal final y trunca a
   siete caracteres, de modo que "vacación" y "vacaciones" producen el mismo término.

### Diferencia con la tokenización de un LLM

Ambos procesos se denominan tokenización, pero tienen fines distintos:

| | Tokenización para búsqueda (este módulo) | Tokenización de un LLM |
|---|---|---|
| Objetivo | Que dos formas de una palabra coincidan | Representar cualquier texto con un vocabulario fijo |
| Unidad | Palabras normalizadas | Subpalabras (algoritmos como BPE) |
| Pérdida de información | Sí, deliberada (tildes, plurales) | No, el texto es reconstruible |
| Uso en RAG | Construir el índice léxico | Determina el coste y la capacidad de contexto del modelo |

Fundamenta no implementa la tokenización de un LLM: la gestiona cada proveedor. Sí la
tiene en cuenta al dimensionar los fragmentos y los prompts.

## BM25

La puntuación de un fragmento para una consulta suma, por cada término de la consulta:

```text
IDF(t) · f · (k1 + 1) / (f + k1 · (1 − b + b · longitud / longitud_media))
```

- **f**: frecuencia del término en el fragmento. Su efecto se satura (`k1 = 1.5`):
  repetir un término muchas veces apenas suma.
- **IDF**: rareza del término en la colección. "NovaTech" aparece en casi todos los
  fragmentos (IDF 0,35); "mudanza", en uno (IDF 3,33).
- **Longitud**: los fragmentos largos se penalizan (`b = 0.75`) para no favorecerlos
  solo por contener más palabras.

## Indexación con contexto de sección

`contextualize` antepone la sección al texto de cada fragmento antes de indexarlo. En
el cuaderno, la consulta "permisos retribuidos por mudanza" sitúa la tabla de permisos
en tercera posición indexando solo el texto y en primera posición indexando la
sección y el texto.

## Fortalezas y limitaciones

BM25 es rápido, no requiere modelos ni entrenamiento y es muy preciso con
identificadores, nombres propios, cifras y terminología exacta. En el conjunto de
evaluación (módulo 09) localiza el fragmento correcto en el 100 % de las preguntas
formuladas con el vocabulario de la documentación.

Su limitación es el vocabulario: la consulta "Me han robado el laptop" no devuelve
ningún resultado, porque ni "robado" ni "laptop" aparecen en los documentos. En las
preguntas parafraseadas la tasa de acierto baja al 50 %. Los embeddings (módulo 04)
resuelven este problema.

## Ejercicios propuestos

1. Comparar los resultados con `stem=False` y medir la diferencia con el módulo 09.
2. Implementar TF-IDF y comparar su ranking con el de BM25.
3. Ajustar `k1` y `b` y observar su efecto en fragmentos de distinta longitud.
4. Añadir un glosario de sinónimos de la organización para expandir las consultas.

---

[← Módulo 02 · Segmentación](../02-segmentacion/README.md) · [Índice](../README.md)
· [Módulo 04 · Embeddings →](../04-embeddings/README.md)
