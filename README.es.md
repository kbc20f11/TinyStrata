# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

**TinyStrata es un motor de inferencia de alta velocidad en desarrollo para modelos MoE de la clase 30B, exclusivo de Windows y accesible solo mediante API.** Está orientado principalmente a PC equipados con tarjetas gráficas de uso común de **8–24 GB** de VRAM y con **32 GB de RAM del sistema o menos**.

## ¿Por qué desarrollamos TinyStrata?

Con el aumento de los precios de los componentes, montar un PC de gama alta está cada vez más fuera del alcance de muchas personas.

Por eso, TinyStrata nació para ejecutar IA localmente a gran velocidad en PC equipados con tarjetas gráficas de uso común de la clase RTX 3060 o superior. Nos centramos en ejecutar modelos MoE relativamente pequeños, de la clase 30B, con la mayor rapidez posible.

Nuestro objetivo principal son los equipos con **8–24 GB de VRAM y 32 GB de RAM del sistema o menos**. Creemos que este intervalo representa las configuraciones más habituales de PC equipados con tarjetas gráficas de uso común.

Basado en Strata, TinyStrata busca mejorar la velocidad de inferencia de Qwen3.6-35B-A3B en entornos con VRAM limitada. Queremos aprovechar de forma eficiente la memoria de la GPU para que puedas trabajar con conversaciones y documentos largos en tu propio PC.

TinyStrata concentra su desarrollo en ofrecer inferencia local rápida en Windows y hacerla accesible mediante una API. Esta orientación guía nuestra implementación y nuestras pruebas.

También buscamos que empezar sea lo más sencillo posible. TinyStrata elige automáticamente los ajustes de inferencia y memoria para tu hardware, de modo que puedas usar la API sin una configuración complicada ni ajustes manuales.

## ¿Por qué no incluye una interfaz de usuario?

Ya existen muchas interfaces y entornos de agentes compatibles con las API para LLM. Como TinyStrata también admite formatos de API habituales, puedes conectar y usar los que prefieras. Si configurar la conexión te resulta tedioso, puedes pedirle a un agente de IA que lo haga por ti. Al dejar la interfaz en manos del software existente, nos centramos en desarrollar el motor de inferencia y la API.

## Modelo en el que nos centramos

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) sigue siendo el modelo predeterminado por su equilibrio entre velocidad, consumo de memoria y calidad de respuesta en PC con tarjetas gráficas de uso común. TinyStrata se centra en modelos MoE pequeños y también ofrece **soporte experimental para LFM2.5-24B-A2B**.

LFM utiliza el [GGUF Q4_K_M oficial de LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (**14.53 GB**), **KV FP16** y hasta **128K de contexto**, sin MTP ni Vision. Usa caché automática de expertos en VRAM, cálculo de expertos en CPU, streaming de pesos durante la entrada y CUDA Graphs; no omite expertos seleccionados. Sus valores predeterminados son `temperature=0.1`, `top_k=50` y `repetition_penalty=1.05`, con el razonamiento siempre activado. Los ajustes de Qwen y LFM se guardan por separado. El motor LFM sigue en validación y no garantiza tokens idénticos a llama.cpp ni entre distintas distribuciones CPU/GPU.

Si falta el ejecutable de LFM, consulta las [instrucciones de compilación manual](README.md#model-focus).

## Cómo instalar

Ejecuta lo siguiente en PowerShell de 64 bits para instalar TinyStrata e iniciar la API en `http://127.0.0.1:18080/v1`:

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Para volver a iniciarlo después de reiniciar Windows, ejecuta este comando o haz doble clic en el mismo archivo `run-api.bat`:

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

También puedes darle a un agente de IA el enlace al [repositorio de TinyStrata en GitHub](https://github.com/kbc20f11/TinyStrata) y encargarle la instalación.

### Cambiar de modelo sin reiniciar

El instalador inicia Qwen de forma predeterminada. Para preparar LFM en el mismo servidor API, inicia la descarga y consulta el progreso:

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"lfm2.5-24b-a2b"}'
Invoke-RestMethod "$api/v1/model/download?model=lfm2.5-24b-a2b"
```

Repite la consulta hasta que `state` sea `ready` y después solicita el cambio explícitamente. Descargar un modelo inactivo lo prepara sin activarlo:

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"lfm2.5-24b-a2b"}'
Invoke-RestMethod "$api/v1/model/switch"
```

Consulta `GET /v1/model/switch` hasta que `state` sea `ready`. Para volver a Qwen, envía la misma solicitud con `{"model":"qwen3.6-35b-a3b"}`. Iniciar una descarga o un cambio devuelve `202`; solicitar el cambio durante una generación devuelve `409`, así que vuelve a intentarlo cuando termine. Las solicitudes de inferencia durante el cambio reciben `503`. El puerto y la autenticación no cambian. La selección dura la sesión actual del servidor; al reiniciar se usa el modelo elegido en el lanzador.

## Velocidad medida — Qwen3.6

Las siguientes mediciones se realizaron el 2026-10-19 en Windows con una RTX PRO 4000 Blackwell SFF Edition (24 GB), un Ryzen 9 7945HX3D y el modelo fijo Unsloth MTP UD-IQ3_XXS. El proyector de visión permaneció cargado y preparado en la GPU; las solicitudes solo contenían texto, sin tiempo de codificación de imágenes. Se usaron los valores predeterminados de la API: razonamiento activado, temperatura 0.7, MTP 3, bloques de entrada de 512 tokens y selección automática de contexto y ubicación de expertos. La ruta híbrida de CUDA Graph estaba desactivada por defecto; los grafos normales para ejecución íntegra en GPU conservaron su configuración predeterminada.

Las solicitudes omitieron los parámetros de muestreo y la semilla. La única modificación a sus valores predeterminados fue `max_tokens=128` como límite de la prueba, en lugar de 32768. La columna FP8 usa la selección automática de KV en esta GPU; para la comparación INT8 solo se forzó el modo KV a INT8.

**Tenemos previsto realizar más adelante mediciones en equipos reales con GPU de 8 GB, 12 GB y 16 GB.** Las filas con memoria reducida de la tabla siguiente se midieron limitando la memoria disponible en la misma GPU de 24 GB; no son pruebas de rendimiento de GPU físicas de 8/12/16 GB.

| Condición de memoria disponible | Capacidad de contexto | Generación con INT8 | Generación con FP8 nativo | Notas |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | Todos los pesos residentes en VRAM |
| Aprox. 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | 8 GiB reservados; cálculo de expertos en CPU y GPU |
| Aprox. 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | 12 GiB reservados; cálculo de expertos en CPU y GPU |
| Aprox. 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | 16 GiB reservados; cálculo de expertos en CPU y GPU |

## ¿Qué proyecto debería usar?

| Tu objetivo y tu hardware | Proyecto |
| --- | --- |
| Quieres inferencia de Qwen3.6-35B-A3B solo mediante API en **Windows**, con **8–24 GB de VRAM y 32 GB de RAM del sistema o menos**, por ejemplo, en un PC equipado con una tarjeta gráfica de uso común. | TinyStrata (**solo Windows**, en desarrollo) |
| Tienes tanto **una GPU de la clase RTX 3060 o superior** como **64 GB de RAM del sistema o más**. | Recomendamos [Strata (proyecto original)](https://github.com/Niko1221/Strata). |
| Quieres una **interfaz de chat incluida**. | Recomendamos [Strata (proyecto original)](https://github.com/Niko1221/Strata). |

## Cómo se reparte la memoria — Qwen3.6

Un modelo MoE contiene muchos **expertos**: redes neuronales más pequeñas que se seleccionan para cada token. TinyStrata mantiene tantos expertos como puede en la memoria de la GPU (VRAM). Durante la generación, puede calcular en la CPU los expertos seleccionados que no están en caché, usando la RAM del sistema. Al procesar lotes de entrada más grandes, puede, en su lugar, transferir progresivamente los pesos de los expertos desde la RAM a través de búferes temporales de la GPU. Todos los expertos seleccionados por el modelo participan en el cálculo.

Los archivos del modelo descargados suman unos **15.0 GB**: **14.1 GB** para el modelo de lenguaje (incluidos los pesos de MTP) y **0.9 GB** para el proyector de visión (`mmproj-F16.gguf`). No es necesario que todo el modelo de lenguaje quepa en la VRAM. La memoria de la GPU también necesita espacio para el codificador de visión y los búferes de procesamiento de imágenes, la **caché KV** de la conversación, los demás pesos del modelo y los búferes de cálculo temporales:

![Cómo reparte TinyStrata la memoria entre la VRAM de la GPU y la RAM del sistema](assets/memory-sharing.png)

Este es un diagrama de distribución en el modo de imágenes GPU predeterminado, no una partición de tamaño fijo. Cada vez que carga el modelo, TinyStrata inicia y prepara el codificador de visión de la GPU antes de cargar el modelo de lenguaje; sus pesos y búferes de trabajo preparados permanecen residentes. Después asigna la caché KV, los demás pesos y los búferes de trabajo, y dimensiona la caché de expertos según la VRAM libre restante. En Windows usa el menor valor entre la memoria libre que informa CUDA y la VRAM dedicada libre que informa el controlador NVIDIA, cuando ambos están disponibles.

El contenido de la caché cambia a medida que se usan distintos expertos; la capacidad de la caché se recalcula cuando se vuelve a cargar el modelo. Otras aplicaciones que se abran más tarde pueden seguir consumiendo VRAM.

La ubicación de expertos busca normalmente un margen de 768 MiB. Como Strata, cuando falta poco para que todos los expertos del modelo principal quepan en VRAM, puede liberar las copias empaquetadas de los pesos Q8, reducir un búfer de puntuaciones de Attention si existe y bajar el margen a 256 MiB; los expertos MTP pueden ejecutarse en la CPU. `--vram-reserve-mib N` fija la reserva explícitamente. Es un objetivo al ubicar los expertos, no una garantía permanente de memoria libre.

Tener más expertos residentes en la GPU suele implicar menos trabajo para la CPU. En modo híbrido, la RAM del sistema también conserva copias de respaldo de los expertos almacenados en la caché de VRAM, por lo que añadir VRAM no reduce el uso de RAM byte por byte. Deja RAM también para el sistema operativo y las demás aplicaciones. Una mayor **capacidad de RAM** permite alojar los datos del modelo, pero la velocidad también depende de la CPU, el ancho de banda de la RAM y la GPU. La «memoria compartida de GPU» de Windows es distinta del cálculo deliberado de expertos en la CPU que realiza TinyStrata.

La caché KV permanece en la VRAM; la RAM del sistema almacena los datos de los expertos para este mecanismo de descarga de trabajo. Por tanto, una conversación larga sigue necesitando suficiente memoria de GPU para su caché KV.

## Ajustes automáticos — Qwen3.6

Las configuraciones nuevas eligen automáticamente KV/Attention y la capacidad de contexto. No necesitas seleccionar una receta para 8 GB, 12 GB o 16 GB.

Una RTX 3060 selecciona INT8. Los entornos compatibles con RTX de las series 40/50 pueden seleccionar FP8 nativo; el motor comprueba los kernels compilados y las capacidades de la GPU, además de la generación del hardware. Si FP8 nativo no está disponible, el modo automático usa INT8. Esto elige una ruta compatible sin ejecutar una prueba de velocidad o calidad al iniciar. **Attention con FP8 nativo sigue siendo experimental.**

El motor nativo estima al iniciar un tamaño de bloque de entrada de 512, 1024, 2048 o 4096 tokens, según la VRAM libre, la capacidad de contexto y si los expertos caben en la GPU. Los bloques mayores pueden reducir las transferencias repetidas de pesos de RAM a GPU durante el procesamiento de la entrada. `run-api.bat --chunk N` fija el tamaño; `run-api.bat --chunk auto` restablece la selección automática nativa.

La capacidad de contexto es el espacio máximo reservado para la entrada y la salida generada en conjunto. Una capacidad mayor usa más memoria KV incluso con una conversación corta, y deja menos VRAM para los expertos. La capacidad tampoco implica la misma velocidad con cualquier longitud de conversación: procesar un documento largo requiere más trabajo. Como referencia, solo KV en INT8 ocupa unos **0.71 GiB a 64K, 1.42 GiB a 128K y 2.84 GiB a 256K**, incluida la capa MTP; los demás pesos y búferes de trabajo requieren espacio adicional. En esta implementación, FP8 nativo usa aproximadamente el mismo almacenamiento KV que INT8; su ventaja es una ruta de cálculo diferente, en lugar de una caché considerablemente más pequeña.

La VRAM libre es la memoria disponible al cargar el modelo, no la cifra impresa en la caja de la GPU. Por ello, otras aplicaciones pueden hacer que se seleccione un contexto menor. Si una recarga reduce la capacidad, la API rechaza una entrada que ya no quepa en vez de truncarla silenciosamente. En Windows, la selección automática de contexto consulta tanto la memoria libre de CUDA como la lectura de todo el dispositivo del controlador de NVIDIA, y usa el menor valor cuando ambas consultas tienen éxito.

La cuantización de los pesos GGUF y la precisión de KV son independientes. El modelo UD-IQ3_XXS seleccionado tiene pesos comprimidos; INT8 o FP8 describe la caché de la conversación y su ruta de Attention. KV de menor precisión puede cambiar las respuestas, por lo que estos modos no prometen una salida idéntica a la de KV en FP32. La implementación conserva todas las posiciones visibles del contexto y todos los expertos seleccionados por el modelo.

Las configuraciones nuevas del lanzador usan MTP 3, que propone varios tokens por adelantado y los verifica con el modelo principal. Puede mejorar la velocidad de generación, pero la ganancia depende de cuántas propuestas se acepten. La verificación busca preservar el comportamiento de decodificación del modelo principal; las diferencias entre las rutas de ejecución numérica pueden aun así cambiar los tokens, de modo que activar MTP no promete una salida idéntica a la obtenida al desactivarlo.

MTP admite tanto la generación voraz como el muestreo basado en temperatura, incluidas las penalizaciones por repetición, frecuencia y presencia. En las solicitudes con muestreo, el modelo principal muestrea cada fila de verificación y solo se acepta el prefijo del borrador que coincide.

Los ajustes explícitos prevalecen sobre las elecciones automáticas, y los ajustes seleccionados manualmente con anterioridad siguen en vigor. Por ejemplo, `run-api.bat --kv-type int8` fija INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restablece la selección automática de KV, contexto y ubicación de expertos.

## Servidor API

**TinyStrata funciona solo mediante API.** Ofrece las API Chat Completions (`/v1/chat/completions`) y Responses (`/v1/responses`) compatibles con OpenAI, y una API Messages (`/v1/messages`) compatible con Anthropic, incluida la generación en streaming. Conecta un cliente de API o una interfaz externa; TinyStrata no incluye una interfaz de chat.

Para Qwen, el razonamiento está activado de forma predeterminada, con `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` y un máximo de **32,768 tokens de salida**, incluido el razonamiento. Los ajustes de cada solicitud tienen prioridad.

La API escucha en `http://127.0.0.1:18080` de forma predeterminada. Para usar otro puerto, ejecuta `run-api.bat --port 18081` desde la carpeta de TinyStrata instalada.

Para Qwen, `run-api.bat --vision gpu` procesa las imágenes en la GPU (valor predeterminado para nuevas configuraciones); `--vision cpu` usa la CPU y `--vision no` desactiva las imágenes. `--vision-tokens N` fija el límite de tokens de imagen (4096 por defecto). Estas opciones se guardan y se mantienen en los siguientes inicios si se omiten.

La API puede iniciarse antes de que se instalen los archivos del modelo. Con Qwen seleccionado, `POST /v1/model/download` con un cuerpo JSON vacío (`{}`) descarga el [modelo de lenguaje fijo Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, incluidos los pesos de MTP) y su [proyector de visión](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), unos **15.0 GB en total**. Verifica las sumas de comprobación SHA-256 de ambos archivos, extrae el tokenizador y los carga. `GET /v1/model/download` informa del progreso; las descargas interrumpidas se pueden reanudar. Repetir la solicitud de descarga reutiliza el trabajo activo, y `POST /v1/model/download/cancel` puede cancelar la transferencia o la verificación. Que la descarga haya finalizado no significa que la inferencia esté lista: espera a que el estado de la descarga sea `ready` y a que `/health` indique `loaded: true` y, si las imágenes están activadas, también `images: true`.

El instalador prepara los ejecutables de los motores y las bibliotecas de ejecución de la GPU, y descarga Qwen y su proyector de visión de forma predeterminada. Las configuraciones existentes conservan el modelo local seleccionado hasta que lo cambies explícitamente.

## Agradecimientos

TinyStrata es un fork independiente de [Strata](https://github.com/Niko1221/Strata) y se apoya ampliamente en su implementación. Gracias a [Niko](https://github.com/Niko1221) y a la comunidad de Strata por el motor y el trabajo que hizo posible TinyStrata.

Gracias también a [Aaron Flin](https://github.com/aflin) y al [proyecto MoMoE](https://github.com/aflin/momoe) por el código de caché KV, atención y expertos cuantizados adaptado en TinyStrata.

## Licencia

MIT. Consulta [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Atribución al proyecto original: [NOTICE](NOTICE). GGML: [licencia MIT](third_party/ggml/LICENSE).
