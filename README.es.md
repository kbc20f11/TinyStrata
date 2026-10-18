# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

**TinyStrata es un motor de inferencia de alta velocidad en desarrollo para modelos MoE de la clase 30B, exclusivo de Windows y accesible solo mediante API.** Está orientado principalmente a PC para juegos convencionales y portátiles para juegos con GPU de **8–24 GB** de VRAM y **32 GB de RAM del sistema o menos**.

## ¿Por qué desarrollamos TinyStrata?

Con el aumento de los precios de los componentes, montar un PC de gama alta está cada vez más fuera del alcance de muchas personas.

Por eso, TinyStrata nació para ejecutar IA localmente a gran velocidad en PC para juegos convencionales y portátiles para juegos con una GPU de la clase RTX 3060 o superior. Nos centramos en ejecutar modelos MoE relativamente pequeños, de la clase 30B, con la mayor rapidez posible.

Nuestro objetivo principal son los equipos con **8–24 GB de VRAM y 32 GB de RAM del sistema o menos**. Creemos que este intervalo representa las configuraciones más habituales de PC para juegos y portátiles para juegos.

Basado en Strata, TinyStrata busca mejorar la velocidad de inferencia de Qwen3.6-35B-A3B en entornos con VRAM limitada. Queremos aprovechar de forma eficiente la memoria de la GPU para que puedas trabajar con conversaciones y documentos largos en tu propio PC.

TinyStrata concentra su desarrollo en ofrecer inferencia local rápida en Windows y hacerla accesible mediante una API. Esta orientación guía nuestra implementación y nuestras pruebas.

También buscamos que empezar sea lo más sencillo posible. TinyStrata elige automáticamente los ajustes de inferencia y memoria para tu hardware, de modo que puedas usar la API sin una configuración complicada ni ajustes manuales.

## ¿Por qué no incluye una interfaz de usuario?

Ya existen muchas interfaces y entornos de agentes compatibles con las API para LLM. Como TinyStrata también admite formatos de API habituales, puedes conectar y usar los que prefieras. Si configurar la conexión te resulta tedioso, puedes pedirle a un agente de IA que lo haga por ti. Al dejar la interfaz en manos del software existente, nos centramos en desarrollar el motor de inferencia y la API.

## Modelo en el que nos centramos

Actualmente, TinyStrata se centra exclusivamente en [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Consideraremos añadir otro modelo cuando un modelo MoE resulte más adecuado que Qwen3.6-35B-A3B para la inferencia local rápida en PC para juegos convencionales y portátiles para juegos.

Consideramos que uno de los problemas es que, mientras avanzan el desarrollo y la presentación de grandes modelos, no aumentan lo suficiente las opciones de modelos MoE de la clase 30B que sean fáciles de usar en PC para juegos convencionales. [Qwen3.6-35B-A3B, publicado el 16 de abril de 2026](https://github.com/QwenLM/Qwen3.8#news), sigue encajando en nuestro caso de uso, que prioriza el equilibrio entre velocidad, bajo consumo de memoria y calidad de las respuestas, pese al tiempo transcurrido desde su publicación. Todavía no hemos encontrado un modelo MoE pequeño que, a nuestro juicio, supere claramente ese equilibrio en los entornos a los que se dirige el proyecto.

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

## Velocidad medida

Las siguientes mediciones se realizaron en Windows con una RTX PRO 4000 Blackwell SFF Edition (24 GB), un Ryzen 9 7945HX3D y el modelo fijo Unsloth MTP UD-IQ3_XXS el 2026-10-17. Se usaron generación voraz, MTP 3, las capacidades de contexto indicadas a continuación, bloques de entrada de 512 tokens, ubicación automática de expertos y la ruta híbrida opcional de CUDA Graph activada.

**Tenemos previsto realizar más adelante mediciones en equipos reales con GPU de 8 GB, 12 GB y 16 GB.** Las filas con memoria reducida de la tabla siguiente se midieron limitando la memoria disponible en la misma GPU de 24 GB; no son pruebas de rendimiento de GPU físicas de 8/12/16 GB.

| Condición de memoria disponible | Capacidad de contexto | Generación con INT8 | Generación con FP8 nativo | Notas |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 121.28 tok/s | 122.73 tok/s | GPU completa disponible; todos los expertos residentes |
| Aprox. 16 GB | 256K | 96.98 tok/s | 112.39 tok/s | 8 GiB ocupados por separado |
| Aprox. 12 GB | 128K | 104.28 tok/s | 107.61 tok/s | 12 GiB ocupados por separado |
| Aprox. 8 GB | 64K | 81.43 tok/s | 86.69 tok/s | 16 GiB ocupados por separado |

Estas son velocidades agregadas de decodificación a partir de tres entradas cortas (34, 46 y 1,599 tokens), cada una con dos generaciones de 128 tokens. Se excluyen la carga y el procesamiento de la entrada. **64K–256K son capacidades de caché asignadas, no longitudes de entrada.** Todas las filas con memoria reducida usaron asignaciones independientes en la misma GPU de 24 GB y realizaron trabajo real de expertos en la CPU.

Las cifras de 24/16 GB corresponden a la ejecución anterior. Los casos añadidos de 12/8 GB usaron la compilación actual y la selección automática del contexto (128K/64K). La capacidad de contexto también afecta a la velocidad, por lo que estas filas muestran distintos ajustes prácticos. La ventaja de FP8 aquí es un resultado medido en estas condiciones, no una garantía general ni una comparación de calidad.

## ¿Qué proyecto debería usar?

| Tu objetivo y tu hardware | Proyecto |
| --- | --- |
| Quieres inferencia de Qwen3.6-35B-A3B solo mediante API en **Windows**, con **8–24 GB de VRAM y 32 GB de RAM del sistema o menos**, por ejemplo, en un PC para juegos convencional o un portátil para juegos. | TinyStrata (**solo Windows**, en desarrollo) |
| Tienes tanto **una GPU de la clase RTX 3060 o superior** como **64 GB de RAM del sistema o más**. | Recomendamos [Strata (proyecto original)](https://github.com/Niko1221/Strata). |
| Quieres una **interfaz de chat incluida**. | Recomendamos [Strata (proyecto original)](https://github.com/Niko1221/Strata). |

**TinyStrata solo admite Windows.** Linux y macOS quedan fuera del alcance de soporte de este proyecto. Estas cifras de RAM del sistema sirven como orientación para elegir un proyecto. Los requisitos reales de memoria dependen del modelo, la cuantización, la longitud del contexto y la VRAM disponible.

## Cómo se reparte la memoria

Un modelo MoE contiene muchos **expertos**: redes neuronales más pequeñas que se seleccionan para cada token. TinyStrata mantiene tantos expertos como puede en la memoria de la GPU (VRAM). Durante la generación, puede calcular en la CPU los expertos seleccionados que no están en caché, usando la RAM del sistema. Al procesar lotes de entrada más grandes, puede, en su lugar, transferir progresivamente los pesos de los expertos desde la RAM a través de búferes temporales de la GPU. Todos los expertos seleccionados por el modelo participan en el cálculo.

Los archivos del modelo descargados suman unos **15.0 GB**: **14.1 GB** para el modelo de lenguaje (incluidos los pesos de MTP) y **0.9 GB** para el proyector de visión (`mmproj-F16.gguf`). No es necesario que todo el modelo de lenguaje quepa en la VRAM. La memoria de la GPU también necesita espacio para el codificador de visión y los búferes de procesamiento de imágenes, la **caché KV** de la conversación, los demás pesos del modelo y los búferes de cálculo temporales:

![Cómo reparte TinyStrata la memoria entre la VRAM de la GPU y la RAM del sistema](assets/memory-sharing.png)

Este es un diagrama de distribución, no una partición de tamaño fijo. Cada vez que carga el modelo, TinyStrata inicia y prepara el codificador de visión de la GPU antes de cargar el modelo de lenguaje. Después asigna la caché KV, los demás pesos y los búferes de trabajo, y ajusta la caché de expertos a la VRAM libre restante, dejando un margen. En Windows, consulta tanto la lectura de memoria libre de CUDA como la lectura de todo el dispositivo del controlador de NVIDIA, y usa el menor valor disponible cuando ambas consultas tienen éxito. Si la asignación falla o deja demasiado poco margen, vuelve a intentarlo con menos expertos.

El contenido de la caché cambia a medida que se usan distintos expertos; la capacidad de la caché se recalcula cuando se vuelve a cargar el modelo. Otras aplicaciones que se abran más tarde pueden seguir consumiendo VRAM. La reserva es un margen disponible al ubicar los expertos, no una garantía permanente de que queden 768 MiB libres.

Tener más expertos residentes en la GPU suele implicar menos trabajo para la CPU. En modo híbrido, la RAM del sistema también conserva copias de respaldo de los expertos almacenados en la caché de VRAM, por lo que añadir VRAM no reduce el uso de RAM byte por byte. Deja RAM también para el sistema operativo y las demás aplicaciones. Una mayor **capacidad de RAM** permite alojar los datos del modelo, pero la velocidad también depende de la CPU, el ancho de banda de la RAM y la GPU. La «memoria compartida de GPU» de Windows es distinta del cálculo deliberado de expertos en la CPU que realiza TinyStrata.

La caché KV permanece en la VRAM; la RAM del sistema almacena los datos de los expertos para este mecanismo de descarga de trabajo. Por tanto, una conversación larga sigue necesitando suficiente memoria de GPU para su caché KV.

## Ajustes automáticos

Las configuraciones nuevas eligen automáticamente KV/Attention y la capacidad de contexto. No necesitas seleccionar una receta para 8 GB, 12 GB o 16 GB.

![Ajustes automáticos según el hardware para Attention, la caché KV y la capacidad de contexto](assets/automatic-settings.png)

Una RTX 3060 selecciona INT8. Los entornos compatibles con RTX de las series 40/50 pueden seleccionar FP8 nativo; el motor comprueba los kernels compilados y las capacidades de la GPU, además de la generación del hardware. Si FP8 nativo no está disponible, el modo automático usa INT8. Esto elige una ruta compatible sin ejecutar una prueba de velocidad o calidad al iniciar. **Attention con FP8 nativo sigue siendo experimental.**

La capacidad de contexto es el espacio máximo reservado para la entrada y la salida generada en conjunto. Una capacidad mayor usa más memoria KV incluso con una conversación corta, y deja menos VRAM para los expertos. La capacidad tampoco implica la misma velocidad con cualquier longitud de conversación: procesar un documento largo requiere más trabajo. Como referencia, solo KV en INT8 ocupa unos **0.71 GiB a 64K, 1.42 GiB a 128K y 2.84 GiB a 256K**, incluida la capa MTP; los demás pesos y búferes de trabajo requieren espacio adicional. En esta implementación, FP8 nativo usa aproximadamente el mismo almacenamiento KV que INT8; su ventaja es una ruta de cálculo diferente, en lugar de una caché considerablemente más pequeña.

La VRAM libre es la memoria disponible al cargar el modelo, no la cifra impresa en la caja de la GPU. Por ello, otras aplicaciones pueden hacer que se seleccione un contexto menor. Si una recarga reduce la capacidad, la API rechaza una entrada que ya no quepa en vez de truncarla silenciosamente.

La cuantización de los pesos GGUF y la precisión de KV son independientes. El modelo UD-IQ3_XXS seleccionado tiene pesos comprimidos; INT8 o FP8 describe la caché de la conversación y su ruta de Attention. KV de menor precisión puede cambiar las respuestas, por lo que estos modos no prometen una salida idéntica a la de KV en FP32. La implementación conserva todas las posiciones visibles del contexto y todos los expertos seleccionados por el modelo.

Las configuraciones nuevas del lanzador usan MTP 3, que propone varios tokens por adelantado y los verifica con el modelo principal. Puede mejorar la velocidad de generación, pero la ganancia depende de cuántas propuestas se acepten. La verificación busca preservar el comportamiento de decodificación del modelo principal; las diferencias entre las rutas de ejecución numérica pueden aun así cambiar los tokens, de modo que activar MTP no promete una salida idéntica a la obtenida al desactivarlo.

MTP admite tanto la generación voraz como el muestreo basado en temperatura, incluidas las penalizaciones por repetición, frecuencia y presencia. En las solicitudes con muestreo, el modelo principal muestrea cada fila de verificación y solo se acepta el prefijo del borrador que coincide.

Los ajustes explícitos prevalecen sobre las elecciones automáticas, y los ajustes seleccionados manualmente con anterioridad siguen en vigor. Por ejemplo, `run-api.bat --kv-type int8` fija INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restablece la selección automática de KV, contexto y ubicación de expertos.

## Servidor API

**TinyStrata funciona solo mediante API.** Ofrece las API Chat Completions (`/v1/chat/completions`) y Responses (`/v1/responses`) compatibles con OpenAI, y una API Messages (`/v1/messages`) compatible con Anthropic, incluida la generación en streaming. Conecta un cliente de API o una interfaz externa; TinyStrata no incluye una interfaz de chat.

De forma predeterminada, el razonamiento está activado, con `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` y un máximo de **32,768 tokens de salida**, incluido el razonamiento. Los ajustes de cada solicitud tienen prioridad.

La API escucha en `http://127.0.0.1:18080` de forma predeterminada. Para usar otro puerto, ejecuta `run-api.bat --port 18081` desde la carpeta de TinyStrata instalada.

La API puede iniciarse antes de que se instalen los archivos del modelo. `POST /v1/model/download` con un cuerpo JSON vacío (`{}`) descarga el [modelo de lenguaje fijo Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, incluidos los pesos de MTP) y su [proyector de visión](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), unos **15.0 GB en total**. Verifica las sumas de comprobación SHA-256 de ambos archivos, extrae el tokenizador y los carga. `GET /v1/model/download` informa del progreso; las descargas interrumpidas se pueden reanudar. Repetir la solicitud de descarga reutiliza el trabajo activo, y `POST /v1/model/download/cancel` puede cancelar la transferencia o la verificación. Que la descarga haya finalizado no significa que la inferencia esté lista: espera a que el estado de la descarga sea `ready` y a que `/health` indique `loaded: true` e `images: true`.

El instalador prepara el motor y las bibliotecas de ejecución de la GPU y descarga automáticamente ambos archivos del modelo. Las configuraciones existentes conservan el modelo local seleccionado hasta que lo cambies explícitamente.

## Agradecimientos

TinyStrata es un fork independiente de [Strata](https://github.com/Niko1221/Strata) y se apoya ampliamente en su implementación. Gracias a [Niko](https://github.com/Niko1221) y a la comunidad de Strata por el motor y el trabajo que hizo posible TinyStrata.

Gracias también a [Aaron Flin](https://github.com/aflin) y al [proyecto MoMoE](https://github.com/aflin/momoe) por el código de caché KV, atención y expertos cuantizados adaptado en TinyStrata.

## Licencia

MIT. Consulta [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Atribución al proyecto original: [NOTICE](NOTICE). GGML: [licencia MIT](third_party/ggml/LICENSE).
