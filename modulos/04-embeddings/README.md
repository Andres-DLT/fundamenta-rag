# Módulo 04 · Embeddings

> Representación del significado de un texto como un vector, de modo que textos con
> significados similares queden próximos aunque no compartan palabras.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/embeddings.py`](../../fundamenta/embeddings.py)
· [pruebas](../../tests/test_search.py)

---

## Objetivos

- Comprender la similitud coseno.
- Implementar embeddings desde cero (LSA) y conocer sus límites.
- Utilizar un modelo neuronal multilingüe que se ejecuta en local.

## Similitud coseno

Un embedding es un vector de números. La similitud entre dos textos se mide con el
coseno del ángulo entre sus vectores: 1 si apuntan en la misma dirección, 0 si no
guardan relación. Si los vectores están normalizados (longitud 1), el coseno coincide
con el producto escalar, lo que permite calcular la similitud con todos los fragmentos
en una sola operación matricial (módulo 05).

## Interfaz común

```python
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
```

Documentos y consultas se codifican por separado porque algunos modelos los tratan de
forma distinta.

## Implementación 1: LSA (`LSAEmbedder`)

El análisis semántico latente construye la matriz TF-IDF de la colección y la reduce
mediante descomposición en valores singulares (SVD). Los términos que suelen aparecer
juntos quedan en direcciones próximas.

- No requiere descargas ni dependencias adicionales a NumPy.
- Se entrena con la propia documentación.
- **Limitación:** solo conoce el vocabulario de la colección y necesita muchos
  documentos para aprender relaciones. Con los 41 fragmentos del caso de estudio no
  aporta mejora sobre BM25: la consulta "Me han robado el laptop" obtiene similitud 0
  con todos los fragmentos.

## Implementación 2: modelo multilingüe (`SentenceTransformerEmbedder`)

Se utiliza `intfloat/multilingual-e5-small`, un modelo preentrenado con millones de
pares de textos en más de 90 idiomas:

- 384 dimensiones, unos 470 MB, ejecución en CPU.
- **Se ejecuta en local**: los documentos de la organización no se envían a terceros.
- Utiliza los prefijos `query:` y `passage:` para distinguir preguntas y documentos;
  la clase los añade automáticamente.

Con este modelo, "Me han robado el laptop" recupera en primera posición la FAQ "¿Qué
hago si pierdo el portátil o me lo roban?". En el conjunto de evaluación, la tasa de
acierto en preguntas parafraseadas pasa del 50 % (BM25) al 92 %.

## Selección del modelo

| Criterio | Recomendación |
|---|---|
| Idioma | Modelos multilingües o específicos del idioma de los documentos |
| Confidencialidad | Modelos locales (como E5) frente a APIs externas |
| Calidad | Consultar clasificaciones públicas (MTEB) y, sobre todo, medir con datos propios |
| Coste y latencia | Modelos pequeños (384 dimensiones) son suficientes para la mayoría de casos |

El sistema admite cualquier modelo compatible con `sentence-transformers` mediante el
parámetro `model_name`.

## Ejercicios propuestos

1. Calcular la matriz de similitud entre las preguntas del conjunto de evaluación.
2. Probar `paraphrase-multilingual-MiniLM-L12-v2` y comparar resultados en el módulo 09.
3. Medir el tiempo de codificación de la colección con distintos tamaños de lote.
4. Implementar un `Embedder` que llame a una API externa de embeddings.

---

[← Módulo 03 · Búsqueda léxica](../03-busqueda-lexica/README.md) · [Índice](../README.md)
· [Módulo 05 · Almacén vectorial →](../05-almacen-vectorial/README.md)
