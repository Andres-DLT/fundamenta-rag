# Módulo 05 · Almacén vectorial

> Almacenamiento de los embeddings y búsqueda por similitud con filtros por metadatos
> y persistencia en disco.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/vector_store.py`](../../fundamenta/vector_store.py)
· [pruebas](../../tests/test_search.py)

---

## Objetivos

- Implementar una búsqueda vectorial exacta eficiente.
- Restringir la búsqueda mediante metadatos.
- Determinar cuándo es necesaria una base de datos vectorial dedicada.

## Implementación

`VectorStore` mantiene una matriz de NumPy con un vector normalizado por fragmento y
la lista de documentos asociados.

```python
scores = self.vectors @ query  # similitud coseno con todos los fragmentos
top = np.argpartition(-scores, k - 1)[:k]
```

- **Una sola operación matricial** calcula la similitud con toda la colección.
- **`argpartition`** obtiene los *k* mejores sin ordenar la colección completa.
- **Validaciones**: un vector por documento y dimensión constante.

## Filtros por metadatos

```python
store.search(query, k=4, where={"source": "productos-y-precios.md"})
```

Los filtros se aplican antes de seleccionar los resultados. En un entorno empresarial
son la base del **control de acceso**: si cada fragmento lleva el departamento o el
nivel de confidencialidad de su documento, la búsqueda solo devuelve lo que el
usuario puede consultar y el modelo nunca recibe información restringida.

## Persistencia

`save` genera `vectors.npy` (matriz) y `documents.json` (texto y metadatos); `load`
los recupera. Calcular los embeddings es la operación más costosa de la indexación,
por lo que persistirlos evita repetirla en cada arranque.

## Rendimiento

Tiempo medio por búsqueda medido en el cuaderno (vectores de 384 dimensiones, CPU de
un equipo de escritorio):

| Fragmentos | Tiempo por búsqueda |
|---:|---:|
| 1.000 | 0,13 ms |
| 10.000 | 0,57 ms |
| 100.000 | 4,5 ms |

La búsqueda exacta es suficiente para la mayoría de bases documentales corporativas.
Las bases de datos vectoriales con índices aproximados (HNSW, IVF) se justifican
cuando:

- la colección supera varios millones de fragmentos;
- se necesitan actualizaciones concurrentes, replicación o alta disponibilidad;
- el control de acceso y la auditoría deben gestionarse en la propia base de datos.

El módulo 11 compara las opciones habituales (pgvector, Qdrant, OpenSearch, entre otras).

## Ejercicios propuestos

1. Añadir un método `delete(where)` que elimine fragmentos por metadatos.
2. Admitir filtros con listas de valores (`{"source": ["a.md", "b.md"]}`).
3. Guardar los vectores en `float16` y medir la pérdida de precisión.
4. Implementar una búsqueda aproximada sencilla por agrupamiento (k-means) y compararla.

---

[← Módulo 04 · Embeddings](../04-embeddings/README.md) · [Índice](../README.md)
· [Módulo 06 · Recuperación híbrida →](../06-recuperacion-hibrida/README.md)
