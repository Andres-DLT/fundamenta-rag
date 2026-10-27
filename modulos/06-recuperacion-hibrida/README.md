# Módulo 06 · Recuperación híbrida

> Combinación de la búsqueda léxica y la semántica mediante Reciprocal Rank Fusion,
> con filtros y control de la diversidad de los resultados.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/retriever.py`](../../fundamenta/retriever.py)
· [pruebas](../../tests/test_search.py)

---

## Objetivos

- Combinar dos rankings con escalas de puntuación distintas.
- Ofrecer una interfaz única de recuperación con tres modos.
- Evitar resultados redundantes.

## Complementariedad de ambos métodos

| | Búsqueda léxica (BM25) | Búsqueda semántica (embeddings) |
|---|---|---|
| Identificadores, códigos, cifras | Muy precisa | Variable |
| Sinónimos y paráfrasis | No los reconoce | Los reconoce |
| Terminología muy específica del sector | Precisa | Depende del modelo |
| Coste | Mínimo | Requiere un modelo |

## Reciprocal Rank Fusion (RRF)

Las puntuaciones de BM25 (sin límite superior) y del coseno (entre -1 y 1) no son
comparables. RRF utiliza solo las posiciones:

```text
puntuación(d) = Σ peso_i / (60 + posición_i(d))
```

Un fragmento que aparece en ambos rankings acumula puntuación de los dos. La constante
60 procede del artículo original (Cormack, Clarke y Büttcher, 2009) y amortigua la
diferencia entre las primeras posiciones.

## La clase `Retriever`

```python
retriever = Retriever(chunks, embedder=SentenceTransformerEmbedder(), mode="hybrid")
results = retriever.search("¿Qué es NovaConnect?", k=4, where={"source": "faq-soporte-it.txt"})
```

| Parámetro | Función | Valor por defecto |
|---|---|---|
| `mode` | `lexical`, `semantic` o `hybrid` | `hybrid` |
| `contextual` | Indexar la sección junto al texto | `True` |
| `candidates` | Resultados de cada método antes de fusionar | 20 |
| `lexical_weight` | Peso de BM25 en la fusión (el semántico pesa 1) | 0,5 |
| `diversity` | Activa MMR (0 = solo relevancia, 1 = solo diversidad) | desactivado |

El peso léxico de 0,5 es el que obtuvo mejor resultado en la evaluación del módulo 10.

## Diversidad: Maximal Marginal Relevance (MMR)

Cuando varios fragmentos casi idénticos ocupan los primeros puestos, el prompt
desperdicia espacio. MMR selecciona en cada paso el candidato que maximiza su
relevancia penalizando su similitud con los ya elegidos. Resulta útil en colecciones
con documentos duplicados o versiones sucesivas de un mismo documento.

## Resultados

En el conjunto de evaluación del módulo 09, con el modelo E5:

| Modo | Acierto@4 | MRR |
|---|---:|---:|
| Léxico (BM25) | 78 % | 0,759 |
| Semántico | 96 % | 0,824 |
| Híbrido | 93 % | 0,824 |

En una colección pequeña y bien redactada, un buen modelo multilingüe obtiene por sí
solo resultados equivalentes al modo híbrido: la diferencia es de una pregunta. El
modo híbrido se mantiene por defecto porque protege frente a los casos en los que los
embeddings son débiles —códigos de producto, referencias normativas, jerga interna,
idiomas poco representados— y porque permite seguir funcionando con BM25 si el modelo
no está disponible. Cada organización debe validar la configuración con sus propias
preguntas (módulo 09).

## Ejercicios propuestos

1. Sustituir RRF por una combinación lineal de puntuaciones normalizadas y comparar.
2. Medir el efecto del parámetro `candidates` en la evaluación.
3. Añadir un filtro por fecha de documento para priorizar las versiones recientes.

---

[← Módulo 05 · Almacén vectorial](../05-almacen-vectorial/README.md) · [Índice](../README.md)
· [Módulo 07 · Generación con citas →](../07-generacion/README.md)
