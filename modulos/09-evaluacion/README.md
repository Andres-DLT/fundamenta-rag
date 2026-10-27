# Módulo 09 · Evaluación

> Medición objetiva de la calidad del sistema con un conjunto de preguntas de
> referencia, como base para cualquier decisión de configuración.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/evaluation.py`](../../fundamenta/evaluation.py)
· [`evaluacion/preguntas.jsonl`](../../evaluacion/preguntas.jsonl) · [pruebas](../../tests/test_application.py)

---

## Objetivos

- Construir un conjunto de preguntas de referencia.
- Medir la recuperación y la calidad de las respuestas por separado.
- Comparar configuraciones con datos en lugar de intuiciones.

## Conjunto de preguntas

[`evaluacion/preguntas.jsonl`](../../evaluacion/preguntas.jsonl) contiene 30 preguntas
sobre la documentación de NovaTech, en tres categorías:

| Categoría | Preguntas | Descripción |
|---|---:|---|
| Directa | 15 | Formuladas con el vocabulario de la documentación |
| Parafraseada | 12 | Formuladas como lo haría un usuario ("Me han robado el laptop") |
| Sin respuesta | 3 | La documentación no contiene la información |

Cada pregunta con respuesta indica el documento de origen y uno o varios **textos de
evidencia**. Un fragmento se considera relevante si procede de ese documento y contiene
todas las evidencias. Este criterio no depende de cómo se haya segmentado el
documento, lo que permite comparar configuraciones distintas de forma justa.

```json
{"id": "P01", "category": "parafraseada", "question": "Me han robado el laptop, ¿qué tengo que hacer?",
 "source": "faq-soporte-it.txt", "evidence": ["bloqueará el equipo"], "answer_contains": ["900 000 111"]}
```

## Métricas de recuperación

- **Tasa de acierto (recall@k):** proporción de preguntas en las que algún fragmento
  relevante está entre los *k* primeros. Indica si el modelo recibirá la información.
- **MRR (Mean Reciprocal Rank):** media de 1/posición del primer fragmento relevante.
  Vale 1 si siempre aparece el primero y premia que la información esté arriba.

## Resultados

27 preguntas con respuesta, *k* = 4, configuración por defecto:

| Configuración | Acierto@4 | MRR | Directas | Parafraseadas |
|---|---:|---:|---:|---:|
| Léxica (BM25) | 78 % | 0,759 | 100 % | 50 % |
| Semántica (LSA) | 78 % | 0,722 | 100 % | 50 % |
| Híbrida (BM25 + LSA) | 78 % | 0,722 | 100 % | 50 % |
| Semántica (E5) | 96 % | 0,824 | 100 % | 92 % |
| **Híbrida (BM25 + E5)** — por defecto | **93 %** | **0,824** | **100 %** | **83 %** |

Conclusiones:

1. **Todas las configuraciones resuelven las preguntas directas.** La diferencia está
   en las preguntas formuladas con otras palabras, que son las habituales en un uso real.
2. **El modelo neuronal multilingüe es el factor determinante**: eleva el acierto en
   preguntas parafraseadas del 50 % al 83-92 %. LSA no aporta mejora con una colección
   de este tamaño.
3. **La búsqueda semántica sola y la híbrida difieren en una pregunta**, una diferencia
   que no es significativa con 27 preguntas. Se mantiene la híbrida por su robustez
   ante identificadores y terminología específica (módulo 06).

### Análisis de errores

La configuración por defecto falla en P03 ("¿Ayuda la empresa con los gastos de
deporte?") y P04 ("¿Puedo trabajar desde casa?"). En P03, la palabra "gastos" hace que
la política de gastos ocupe las primeras posiciones, por delante del beneficio de
gimnasio. Este tipo de análisis orienta las mejoras del módulo 10.

## Evaluación de respuestas

Requiere un LLM y supone una llamada por pregunta:

```bash
fundamenta eval --embedder semantic --llm claude --answers
```

Se aplican dos criterios automáticos:

- **Exactitud:** la respuesta contiene los datos esperados (`answer_contains`).
- **Abstención:** en las preguntas sin respuesta, el sistema utiliza el texto de
  abstención en lugar de inventar.

Además, `judge_faithfulness` emplea un LLM como evaluador para verificar que cada
afirmación de la respuesta está respaldada por las fuentes citadas, lo que detecta
alucinaciones que una comprobación por palabras clave no identifica.

## Recomendaciones para un proyecto real

- Reunir entre 50 y 200 preguntas reales de los usuarios, no solo redactadas por el
  equipo técnico.
- Incluir preguntas sin respuesta y preguntas que combinen varios documentos.
- Ejecutar la evaluación tras cada cambio de documentos, modelo o configuración, y
  conservar el histórico de resultados.

## Ejercicios propuestos

1. Añadir 10 preguntas nuevas, incluida alguna que requiera dos documentos.
2. Calcular el intervalo de confianza de la tasa de acierto.
3. Evaluar las respuestas con un LLM y analizar los fallos de abstención.

---

[← Módulo 08 · Pipeline y demostración](../08-pipeline-y-demo/README.md) · [Índice](../README.md)
· [Módulo 10 · Mejoras de calidad →](../10-mejoras/README.md)
