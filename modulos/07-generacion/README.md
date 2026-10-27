# Módulo 07 · Generación con citas

> Construcción del prompt, conexión con distintos proveedores de modelos de lenguaje y
> vinculación de cada afirmación con su fuente.

**Material:** [cuaderno](notebook.ipynb) · [`fundamenta/generation.py`](../../fundamenta/generation.py)
· [`fundamenta/llm.py`](../../fundamenta/llm.py) · [pruebas](../../tests/test_generation.py)

---

## Objetivos

- Diseñar instrucciones que limiten el modelo a las fuentes proporcionadas.
- Desacoplar el sistema del proveedor del modelo.
- Ofrecer respuestas verificables mediante citas.

## Instrucciones del sistema

El prompt del sistema establece cuatro normas:

1. Responder **exclusivamente** con la información de los fragmentos.
2. **Citar** cada afirmación con el número del fragmento: `[1]`, `[2][3]`.
3. No utilizar conocimientos externos.
4. Si la información no está disponible, responder con un texto fijo de abstención.

La abstención explícita es esencial en un entorno corporativo: una respuesta
"no dispongo de información" es preferible a una respuesta plausible pero incorrecta.
Al ser un texto fijo, el sistema puede detectarla automáticamente y medirla (módulo 09).

## Estructura del prompt

```text
Fragmentos de documentación:

[1] Fuente: manual-del-empleado.md › Vacaciones
## Vacaciones
Cada persona tiene 23 días laborables de vacaciones al año…

[2] Fuente: beneficios-2026.pdf › página 2
…

Pregunta: ¿Cuántos días de vacaciones tengo y cómo se solicitan?
```

Cada fragmento lleva un número y una referencia legible. Tras la respuesta, el sistema
identifica los números citados y conserva como fuentes únicamente esos fragmentos.

## Proveedores

Todos implementan la misma interfaz, `generate(system, prompt) -> str`:

| Proveedor | Clase | Características |
|---|---|---|
| Claude (Anthropic) | `ClaudeLLM` | SDK oficial; modelo `claude-opus-5-5` por defecto; respaldo automático del servidor si el modelo declina una petición |
| Modelos abiertos en local | `OllamaLLM` | Cualquier modelo servido por Ollama; los datos no salen de la infraestructura |
| Sin LLM | modo extractivo | Devuelve los fragmentos relevantes con sus referencias |

```python
from fundamenta.llm import ClaudeLLM, OllamaLLM

llm = ClaudeLLM()  # credenciales en ANTHROPIC_API_KEY
llm = ClaudeLLM(model="claude-sonnet-5-5", effort="low")
llm = OllamaLLM(model="qwen2.5:7b")
```

`ClaudeLLM` admite el parámetro `effort` (`low`, `medium`, `high`), que equilibra
profundidad de razonamiento y coste. Para preguntas documentales directas, `low` o
`medium` suelen ser suficientes; conviene validarlo con el módulo 09.

## Modo extractivo

Sin LLM configurado, el sistema devuelve los fragmentos más relevantes con sus
referencias. Permite realizar demostraciones sin credenciales, operar en entornos sin
acceso a proveedores externos y disponer de una alternativa si el proveedor falla.

## Consideraciones de seguridad

- **Inyección de instrucciones.** Un documento indexado puede contener texto que
  intente modificar el comportamiento del modelo. Las instrucciones del sistema
  prevalecen, pero conviene indexar solo fuentes de confianza y revisar las de origen
  externo.
- **Confidencialidad.** Con `ClaudeLLM`, los fragmentos recuperados se envían al
  proveedor. Si la política de datos no lo permite, `OllamaLLM` mantiene todo el
  proceso en la infraestructura propia.

## Ejercicios propuestos

1. Añadir una instrucción para que el modelo indique la fecha de vigencia del documento.
2. Implementar un proveedor adicional con la misma interfaz.
3. Devolver la respuesta en formato estructurado (JSON) con la lista de citas.

---

[← Módulo 06 · Recuperación híbrida](../06-recuperacion-hibrida/README.md) · [Índice](../README.md)
· [Módulo 08 · Pipeline y demostración →](../08-pipeline-y-demo/README.md)
