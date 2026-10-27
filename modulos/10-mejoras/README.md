# Módulo 10 · Mejoras de calidad

> Ajuste de los parámetros del sistema validando cada cambio con el conjunto de
> evaluación, y técnicas adicionales para casos difíciles.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/query.py`](../../fundamenta/query.py)
· [`fundamenta/retriever.py`](../../fundamenta/retriever.py)

---

## Metodología

Cada ajuste se mide con las 27 preguntas con respuesta del módulo 09, sobre la
recuperación híbrida con el modelo E5. Con este tamaño de muestra, cada pregunta
equivale a 3,7 puntos de tasa de acierto, por lo que las diferencias de una pregunta
deben interpretarse con cautela.

## 1. Contexto de sección en el índice

| | Acierto@4 | MRR |
|---|---:|---:|
| Léxica, solo texto | 78 % | 0,741 |
| Léxica, sección + texto | 78 % | 0,759 |
| Híbrida, solo texto | 89 % | 0,846 |
| Híbrida, sección + texto | **93 %** | 0,824 |

Indexar la sección mejora la posición de los fragmentos que dependen de su título
(como la tabla de permisos) y aumenta la tasa de acierto de la búsqueda híbrida. Se
mantiene activado.

## 2. Peso de la búsqueda léxica en la fusión

| Peso léxico | Acierto@4 | MRR |
|---:|---:|---:|
| 1,0 | 89 % | 0,812 |
| 0,7 | 93 % | 0,821 |
| **0,5** | **93 %** | **0,824** |
| 0,3 | 93 % | 0,806 |
| Solo semántica | 96 % | 0,824 |

Con el modelo E5, reducir el peso de BM25 evita que coincidencias de palabras poco
significativas desplacen resultados semánticamente correctos. El valor 0,5 es el
óptimo entre las configuraciones híbridas y se adopta por defecto.

## 3. Tamaño de fragmento

| Tamaño | Fragmentos | Acierto@4 | MRR |
|---:|---:|---:|---:|
| 300 | 61 | 93 % | 0,818 |
| 500 | 41 | 93 % | 0,824 |
| 800 | 32 | 93 % | 0,824 |

En esta colección el tamaño apenas influye, porque las secciones de los documentos ya
son breves. En colecciones con documentos largos y poco estructurados el efecto es
mayor y debe medirse.

## 4. Número de fragmentos recuperados

| *k* | Acierto@k | MRR |
|---:|---:|---:|
| 1 | 78 % | 0,778 |
| 2 | 78 % | 0,778 |
| **4** | **93 %** | **0,824** |
| 6 | 93 % | 0,824 |
| 8 | 96 % | 0,829 |

Pasar de 2 a 4 fragmentos aporta una mejora clara; a partir de 4, la ganancia es
marginal y el prompt duplica su tamaño. Se adopta *k* = 4.

## 5. Reescritura de consultas

Con `PipelineConfig(rewrite_queries=True)`, el LLM reformula la pregunta con el
vocabulario formal de la documentación antes de buscar. En el cuaderno, la pregunta
"¿Ayuda la empresa con los gastos de deporte?" —uno de los fallos del módulo 09—
recupera el beneficio de gimnasio cuando se reformula como "reembolso cuota gimnasio
actividad deportiva bienestar beneficios".

Supone una llamada adicional al modelo por pregunta, con su coste y latencia. Se
recomienda activarla solo si la evaluación con un LLM confirma la mejora en el caso de
uso concreto.

## 6. Otras técnicas

| Técnica | Descripción | Cuándo aplicarla |
|---|---|---|
| Reordenación (*reranking*) | Un modelo *cross-encoder* puntúa cada par pregunta-fragmento | Cuando la respuesta está entre los 20 primeros pero no entre los 4 primeros |
| Diversidad (MMR) | Evita fragmentos redundantes (módulo 06) | Colecciones con duplicados o versiones |
| Glosario corporativo | Expande las consultas con sinónimos internos | Jerga, siglas y nombres de producto |
| Preguntas hipotéticas | Indexar preguntas generadas para cada fragmento | Documentación técnica con vocabulario muy distinto al de los usuarios |

## Configuración resultante

```python
PipelineConfig(chunk_size=500, overlap=50, k=4, mode="hybrid", contextual=True, lexical_weight=0.5)
```

Esta configuración es un punto de partida sólido, no un óptimo universal: en cada
implantación debe repetirse la evaluación con preguntas reales de la organización.

---

[← Módulo 09 · Evaluación](../09-evaluacion/README.md) · [Índice](../README.md)
· [Módulo 11 · Puesta en producción →](../11-produccion/README.md)
