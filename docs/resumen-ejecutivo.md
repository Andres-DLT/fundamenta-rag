# Fundamenta · Resumen ejecutivo

**Un asistente que responde con la documentación de su organización y cita sus fuentes.**

---

## El reto

El conocimiento de una organización está repartido entre manuales, políticas,
procedimientos, catálogos y bases de conocimiento. Localizar una respuesta concreta
exige saber dónde buscar y leer documentos completos. En consecuencia:

- los equipos dedican tiempo a buscar información que ya existe;
- las mismas consultas llegan una y otra vez a los departamentos de Personas, IT,
  Finanzas o Soporte;
- las respuestas varían según quién las dé y pueden basarse en versiones desactualizadas.

Los asistentes de IA generalistas no resuelven este problema: desconocen la
documentación interna y, cuando no saben algo, pueden ofrecer respuestas verosímiles
pero incorrectas.

## La solución

Fundamenta conecta un modelo de lenguaje con la documentación de la organización
mediante la técnica RAG (*Retrieval-Augmented Generation*):

1. **Indexa** los documentos de la organización (PDF, Markdown, texto).
2. **Recupera**, para cada pregunta, los fragmentos relevantes, aunque la pregunta use
   palabras distintas a las del documento.
3. **Genera** una respuesta basada exclusivamente en esos fragmentos, con una cita a
   la fuente de cada afirmación.

Si la documentación no contiene la respuesta, el sistema lo indica expresamente.

## Beneficios

| Beneficio | Cómo se consigue |
|---|---|
| **Respuestas verificables** | Cada afirmación enlaza con el documento, la sección o la página de origen. |
| **Menos respuestas inventadas** | El modelo solo puede usar los fragmentos recuperados y debe abstenerse si no encuentra la información. |
| **Coherencia** | Todas las personas reciben la misma respuesta, basada en la documentación vigente. |
| **Confidencialidad** | La búsqueda semántica se ejecuta en local; la generación puede hacerse con modelos en infraestructura propia. |
| **Control de acceso** | Cada usuario solo obtiene respuestas de los documentos que tiene autorizados. |
| **Sin dependencia de proveedor** | Compatible con Claude (Anthropic) y con modelos abiertos; el proveedor puede cambiarse sin rediseñar el sistema. |
| **Calidad medible** | Evaluación automática con preguntas de referencia antes y después de cada cambio. |

## Casos de uso

- **Asistente interno para la plantilla**: vacaciones, permisos, gastos, beneficios,
  procedimientos de IT.
- **Soporte al cliente**: respuestas sobre productos, tarifas y condiciones a partir de
  la documentación comercial.
- **Equipos técnicos y de operaciones**: consulta de manuales, procedimientos y
  documentación de sistemas.
- **Áreas legal y de cumplimiento**: localización de cláusulas y políticas con
  referencia a su origen.

## Resultados en el caso de demostración

Sobre la documentación de una empresa ficticia (manual del empleado, política de
gastos, catálogo de productos, FAQ de IT y documento de beneficios):

| | Búsqueda tradicional por palabras | Fundamenta |
|---|---:|---:|
| Preguntas formuladas con el vocabulario de los documentos | 100 % | 100 % |
| Preguntas formuladas con otras palabras | 50 % | 83 % |
| Total | 78 % | 93 % |

*Porcentaje de preguntas en las que el fragmento con la respuesta se encuentra entre los
cuatro primeros resultados. Conjunto de 27 preguntas con respuesta. En cada implantación
se repite la evaluación con preguntas reales de la organización.*

## Modalidades de colaboración

| Modalidad | Alcance |
|---|---|
| **Prueba de concepto** | Indexación de una muestra de la documentación, conjunto de preguntas de evaluación propio y demostración funcional con los resultados medidos. |
| **Implantación** | Integración con las fuentes documentales, control de acceso, despliegue en la infraestructura elegida, monitorización y evaluación continua. |
| **Formación técnica** | Impartición del programa de doce módulos al equipo técnico de la organización, con ejercicios sobre su propia documentación. |

Alcance, plazos y condiciones se definen con cada organización.

## Plan de implantación orientativo

1. **Análisis**: fuentes documentales, perfiles de usuario, requisitos de seguridad y
   selección del proveedor del modelo.
2. **Prueba de concepto**: indexación de una muestra, evaluación con preguntas reales y
   validación de resultados con los responsables del área.
3. **Despliegue**: conexión con las fuentes, autenticación corporativa, control de
   acceso y puesta en marcha con un grupo piloto.
4. **Operación**: ampliación progresiva, incorporación de la valoración de los usuarios
   y evaluación periódica de la calidad.

## Requisitos técnicos

- Servidor con Python 3.11 o superior. La búsqueda semántica funciona en CPU.
- Acceso a un proveedor de modelos de lenguaje (por ejemplo, la API de Anthropic) o a
  un servidor propio con modelos abiertos.
- Acceso de lectura a las fuentes documentales que se vayan a indexar.

## Preguntas frecuentes

**¿Salen nuestros documentos de la organización?**
La indexación y la búsqueda se realizan en la infraestructura propia. Si se utiliza un
proveedor externo para la generación, solo se envían los fragmentos recuperados para
cada pregunta. Con modelos en infraestructura propia, ningún dato sale de ella.

**¿Qué ocurre si la respuesta no está en la documentación?**
El sistema responde que no dispone de información suficiente, en lugar de elaborar una
respuesta sin respaldo.

**¿Funciona en español y en otros idiomas?**
Sí. El modelo de búsqueda semántica es multilingüe y la tokenización léxica está
adaptada al español.

**¿Cómo se mantiene actualizado?**
Los documentos se reindexan cuando cambian. Las respuestas se basan siempre en la
versión indexada.

**¿Cómo sabemos que funciona bien con nuestros documentos?**
Cada implantación incluye un conjunto de preguntas de evaluación propio y la medición
de resultados antes de la puesta en producción.

---

**Andrés de la Torre Flores** · Autor de Fundamenta

Más información técnica: [repositorio del proyecto](https://github.com/Andres-DLT/fundamenta-rag)
