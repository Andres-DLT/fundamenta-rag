# Despliegue

Fundamenta puede desplegarse de dos formas, con la misma aplicación web y la misma
calidad de búsqueda:

| | Configuración ligera (`onnx`) | Imagen Docker completa (`semantic`) |
|---|---|---|
| Modelo de embeddings | E5 multilingüe cuantizado a 8 bits, con ONNX Runtime | E5 multilingüe original, con PyTorch |
| Memoria necesaria | 512 MB (uso real: unos 300 MB) | 2 GB |
| Acierto en la evaluación | 93 % (MRR 0,80) | 93 % (MRR 0,82) |
| Uso recomendado | Demostraciones y planes gratuitos | Producción y desarrollo |

En ambos casos no se necesita GPU.

## Configuración

| Variable | Descripción | Por defecto |
|---|---|---|
| `FUNDAMENTA_PASSWORD` | Contraseña de acceso a la interfaz y a la API. **Recomendada** en cualquier despliegue público. | sin contraseña |
| `FUNDAMENTA_LLM` | `none` (modo extractivo), `claude`, `claude:<modelo>` u `ollama:<modelo>` | `none` |
| `ANTHROPIC_API_KEY` | Clave de la API de Anthropic, necesaria con `FUNDAMENTA_LLM=claude` | — |
| `OLLAMA_HOST` | Dirección del servidor de Ollama, con `FUNDAMENTA_LLM=ollama:<modelo>` | `http://localhost:11434` |
| `FUNDAMENTA_RATE_LIMIT` | Preguntas por minuto y cliente | 30 |
| `FUNDAMENTA_DATA` | Carpeta de documentos dentro del contenedor | `data/novatech` |
| `FUNDAMENTA_EMBEDDER` | `semantic` (modelo E5), `onnx` (E5 cuantizado, sin PyTorch) o `lsa` | `semantic` |
| `PORT` | Puerto HTTP | 7860 |

Las claves y contraseñas deben configurarse como **secretos** de la plataforma, nunca
en el código ni en la imagen.

## Opción 1: Render (plan gratuito)

El repositorio incluye `render.yaml`, que describe el servicio completo: instalación,
descarga del modelo, arranque y comprobación de salud. Render lo publica en una
dirección estable del tipo `https://fundamenta-xxxx.onrender.com`.

1. Cree una cuenta en [render.com](https://render.com) con su cuenta de GitHub.
2. En el panel, elija *New > Blueprint* y seleccione el repositorio `fundamenta-rag`.
3. Render mostrará el servicio `fundamenta` y le pedirá el valor de
   `FUNDAMENTA_PASSWORD`: escriba la contraseña que compartirá con las personas
   invitadas.
4. Pulse *Deploy Blueprint*. La primera construcción tarda unos cinco minutos; cuando el
   estado pase a *Live*, abra la dirección del servicio. El navegador solicitará un
   usuario —cualquiera es válido— y la contraseña configurada.

Cada vez que se actualice la rama principal del repositorio, Render vuelve a desplegar
automáticamente. Para respuestas generadas con Claude, añada en *Environment* las
variables `FUNDAMENTA_LLM=claude` y `ANTHROPIC_API_KEY`.

Los servicios gratuitos de Render se suspenden tras 15 minutos sin uso y se reactivan
con la siguiente visita, que tarda aproximadamente un minuto. Para una demostración
importante, conviene abrir la aplicación unos minutos antes.

## Opción 2: Hugging Face Spaces

Ejecuta la imagen Docker completa con 2 vCPU y 16 GB de memoria, y publica la aplicación
en una dirección estable del tipo `https://<usuario>-fundamenta.hf.space`. Alojar
aplicaciones Docker en Hugging Face requiere una suscripción PRO.

1. Cree una cuenta en [huggingface.co](https://huggingface.co) y un token de acceso
   con permiso de escritura (*Settings > Access Tokens*).
2. Inicie sesión desde la terminal:

   ```bash
   pip install -e ".[deploy]"
   hf auth login
   ```

3. Despliegue:

   ```bash
   python scripts/deploy_huggingface.py --space <usuario>/fundamenta
   ```

   Para respuestas generadas con Claude, añada `--llm claude`.

4. En la página del Space, *Settings > Variables and secrets*, cree los secretos:
   - `FUNDAMENTA_PASSWORD`: la contraseña que compartirá con las personas invitadas.
   - `ANTHROPIC_API_KEY`: solo si utiliza `--llm claude`.

5. Espere a que el estado del Space pase a *Running* (la primera construcción tarda
   varios minutos) y abra la dirección de la aplicación. El navegador solicitará un
   usuario —cualquiera es válido— y la contraseña configurada.

## Opción 3: cualquier plataforma de contenedores

Railway, Google Cloud Run, Azure Container Apps o un servidor propio:

```bash
docker build -t fundamenta .
docker run -p 7860:7860 \
  -e FUNDAMENTA_PASSWORD=... \
  -e FUNDAMENTA_LLM=claude \
  -e ANTHROPIC_API_KEY=... \
  fundamenta
```

Elija una instancia de al menos 2 GB de memoria y configure las variables de la tabla
anterior como secretos de la plataforma.

## Documentación propia

Para desplegar con la documentación de otra organización, sustituya el contenido de
`data/` antes de construir la imagen, o monte un volumen y ajuste `FUNDAMENTA_DATA`:

```bash
docker run -p 7860:7860 -v /ruta/documentos:/datos -e FUNDAMENTA_DATA=/datos fundamenta
```

## Seguridad

- Active siempre `FUNDAMENTA_PASSWORD` en despliegues públicos: sin ella, cualquier
  persona con la dirección puede hacer preguntas y consumir el crédito del proveedor
  del modelo.
- El límite por cliente (`FUNDAMENTA_RATE_LIMIT`) acota el consumo, pero no sustituye a
  la contraseña.
- Con `FUNDAMENTA_LLM=claude`, los fragmentos recuperados se envían a Anthropic. Si la
  documentación es confidencial, utilice Ollama en la propia infraestructura o el modo
  extractivo.
- Para un despliegue corporativo, consulte los requisitos del
  [módulo 11](../modulos/11-produccion/README.md): autenticación con el proveedor de
  identidad, control de acceso por documento y registro de consultas.
