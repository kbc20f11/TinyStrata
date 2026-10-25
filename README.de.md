# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata ist eine schnelle Inferenz-Engine für MoE-Modelle der 30B-Klasse, die ausschließlich für Windows und die Nutzung per API entwickelt wird.** Sie richtet sich vor allem an PCs, die mit gängigen Grafikkarten mit **8–24 GB VRAM** ausgestattet sind und über **höchstens 32 GB Arbeitsspeicher** verfügen.

**NVIDIA-GPUs (CUDA) werden unterstützt. Ein experimenteller AMD-HIP/ROCm-Pfad verwendet INT8-KV; die Prüfung auf AMD-Hardware steht noch aus.**

## Warum wir TinyStrata entwickeln

Angesichts steigender Preise für PC-Komponenten wird es für viele Menschen immer schwieriger, einen High-End-PC zusammenzustellen.

Deshalb wurde TinyStrata entwickelt, um KI lokal mit hoher Geschwindigkeit auf PCs mit gängigen Grafikkarten der RTX 3060-Klasse oder besser auszuführen. Unser Schwerpunkt liegt darauf, vergleichsweise kleine MoE-Modelle der 30B-Klasse so schnell wie möglich auszuführen.

Unsere Hauptzielgruppe sind PCs mit **8–24 GB VRAM und höchstens 32 GB Arbeitsspeicher**. Wir glauben, dass dieser Bereich die häufigsten Konfigurationen von PCs mit gängigen Grafikkarten abdeckt.

TinyStrata baut auf Strata auf und nutzt GPU-Speicher effizient, damit MoE-Modelle der 30B-Klasse auch mit begrenztem VRAM schnell laufen können. Unser Ziel ist es, lange Unterhaltungen und Dokumente auf Ihrem eigenen PC nutzbar zu machen.

Die Entwicklung konzentriert sich auf lokale Inferenz unter Windows und deren Bereitstellung über eine API.

TinyStrata wählt Inferenz- und Speichereinstellungen automatisch passend zu Ihrer Hardware, sodass Sie die API ohne komplizierte Konfiguration oder manuelle Feinabstimmung nutzen können.

## Warum keine integrierte Benutzeroberfläche?

Es gibt bereits viele Benutzeroberflächen und Agenten-Laufzeitumgebungen, die APIs für LLMs unterstützen. Da TinyStrata gängige API-Formate unterstützt, können Sie die Lösung Ihrer Wahl verbinden und nutzen. Falls Ihnen die Einrichtung der Verbindung zu aufwendig erscheint, können Sie einen KI-Agenten damit beauftragen. Indem wir die Benutzeroberfläche bestehender Software überlassen, können wir uns auf die Entwicklung der Inferenz-Engine und der API konzentrieren.

## Modellschwerpunkt

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) bleibt das Standardmodell und bietet auf PCs mit gängigen Grafikkarten ein ausgewogenes Verhältnis von Geschwindigkeit, Speicherbedarf und Antwortqualität. TinyStrata konzentriert sich auf kompakte MoE-Modelle und unterstützt zusätzlich **Gemma 4 26B-A4B-it und LFM2.5-24B-A2B**.

**Qwen3.6-35B-A3B UD-IQ3_XXS** verwendet den festgelegten [Unsloth-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf). Die zwei Downloads umfassen zusammen etwa **15.0 GB**: Sprachmodell einschließlich MTP-Gewichten **14.1 GB** und Vision-Projektor (`mmproj-F16.gguf`) **0.9 GB**.

**Gemma 4 26B-A4B-it QAT Q4_0** verwendet [Googles offizielles Modell und den Vision-Projektor](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) mit dem [passenden Unsloth-QAT-MTP-Assistenten](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Die drei Downloads umfassen **15.886 GB**: Modell **14.439 GB**, Projektor **1.195 GB** und Assistent **0.252 GB**.

**LFM2.5-24B-A2B Q4_K_M** verwendet den offiziellen [LiquidAI-GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF). Der einzelne Sprachmodell-Download ist **14.53 GB** groß.

Alle drei Engines verwenden einen automatischen VRAM-Expertencache, CPU-Expertenberechnung aus dem Arbeitsspeicher, Gewichtsstreaming bei der Eingabe und CUDA Graphs. Sie nutzen dieselbe API und unterstützen Modellwechsel im laufenden Betrieb; die Einstellungen werden pro Modell gespeichert. Ausgewählte Experten werden nicht ausgelassen. Sie unterstützen außerdem automatisches Wachstum des physischen KV-Speichers, ohne die maximale Kontextkapazität zu ändern.

| Funktion | [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B-it) | [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| --- | --- | --- | --- |
| Maximaler Kontext | 256K | 256K | 128K |
| Bildeingabe | Ja; Bildencoder standardmäßig auf der GPU | Ja; Bildencoder standardmäßig auf der GPU | Nein |
| MTP | 3 | 3 | Nein |
| KV-Formate | FP32 / INT8 / natives FP8 | FP16 / INT8 / natives FP8 (experimentell) | FP16 / INT8 / natives FP8 |
| Automatische KV-Auswahl (Standard) | FP8 → INT8 | INT8 | FP8 → INT8 |

Die Spaltenüberschriften verlinken auf die offiziellen Modellkarten. **FP8 → INT8** bedeutet FP8-KV und native FP8-Tensor-Core-Attention, wenn GPU und kompilierte Kernel dies unterstützen, andernfalls INT8. KV-Präzision und GGUF-Gewichtsquantisierung sind unabhängig; niedriger präzises KV kann die Ausgabe gegenüber FP32 (Qwen) beziehungsweise FP16 (Gemma/LFM) verändern.

Gemma verwendet standardmäßig INT8. Natives FP8 ist experimentell und kann mit `run-api.bat --model-family gemma4 --kv-type fp8-mma` ausdrücklich gewählt werden.

Die **automatische Kontextkapazität** richtet sich wie unten gezeigt nach dem gesamten VRAM der GPU; der freie VRAM bestimmt die Größe des Expertencaches. Ein explizites `--max-context N` hat Vorrang.

| Gesamter VRAM (Beispiele) | Qwen3.6 / Gemma 4 | LFM2.5 |
| --- | ---: | ---: |
| 8 GB | 64K | 64K |
| 12 GB | 128K | 128K |
| 16 GB oder mehr | 256K | 128K |

Die Kapazitätsklassen berücksichtigen Unterschiede bei den Treiberangaben, sodass eine 16-GB-Karte auch bei gemeldeten 15.x GiB für Qwen und Gemma 256K auswählt.

Fehlt die Programmdatei eines zusätzlichen Modells, siehe die [Anleitung zum manuellen Build](README.md#model-focus).

## Installation

Führen Sie diesen Befehl in 64-Bit-PowerShell aus, um TinyStrata zu installieren und die API unter `http://127.0.0.1:18080/v1` zu starten:

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Um TinyStrata nach einem Windows-Neustart erneut zu starten, führen Sie diesen Befehl aus oder doppelklicken Sie auf dieselbe `run-api.bat`:

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

Alternativ können Sie einem KI-Agenten den Link zum [GitHub-Repository von TinyStrata](https://github.com/kbc20f11/TinyStrata) geben und ihn mit der Einrichtung beauftragen.

### Modell ohne Neustart wechseln

Das Installationsprogramm startet standardmäßig Qwen. Laden Sie Gemma für denselben laufenden API-Server herunter und prüfen Sie den Fortschritt:

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

Wiederholen Sie die Fortschrittsabfrage bis `state` den Wert `ready` hat und wechseln Sie dann ausdrücklich. Der Download eines inaktiven Modells bereitet es nur vor und aktiviert es nicht:

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

Mit `run-api.bat --model-family gemma4` starten Sie direkt mit Gemma unter `http://127.0.0.1:18082/v1`. Für LFM ersetzen Sie `gemma-4-26b-a4b-it` in den obigen Download-, Fortschritts- und Wechselanfragen durch `lfm2.5-24b-a2b`; mit `run-api.bat --model-family lfm2` starten Sie LFM direkt unter `http://127.0.0.1:18081/v1`.

Fragen Sie `GET /v1/model/switch` bis `state: ready` ab. Zum Zurückwechseln senden Sie dieselbe Wechselanfrage mit `{"model":"qwen3.6-35b-a3b"}`. Der Start eines Downloads oder Wechsels liefert `202`; ein Wechsel während einer Generierung liefert `409`, daher nach deren Ende erneut versuchen. Inferenzanfragen während des Wechsels liefern `503`. Port und Authentifizierung bleiben unverändert. Die Auswahl gilt für die aktuelle Serversitzung; nach einem Neustart wird das im Launcher gewählte Modell verwendet.

## Gemessene Geschwindigkeit — Qwen3.6

Gemessen am **2026-10-21** unter Windows mit einer RTX PRO 4000 Blackwell SFF Edition (24 GB), einem Ryzen 9 7945HX3D, 64 GB Arbeitsspeicher und dem festgelegten Unsloth-MTP-UD-IQ3_XXS-Modell. Der Bildencoder blieb bei den Textanfragen über die API auf der GPU. Thinking, die Sampling-Standardwerte des Modells (Temperatur 0.7) und MTP 3 waren aktiv. Prompt-Chunks und Expertenplatzierung wurden automatisch gewählt; die Kontextkapazität wurde entsprechend der VRAM-Klasse festgelegt. Der hybride CUDA-Graph-Pfad blieb standardmäßig aus, die normalen CUDA Graphs bei vollständigem GPU-Betrieb blieben aktiv.

| Verfügbarer Speicher | Kontextkapazität | INT8-Generierung | Native FP8-Generierung | Anmerkungen |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 117.80 tok/s | 131.34 tok/s | Alle Gewichte im VRAM |
| ca. 16 GB | 256K | 98.90 tok/s | 101.68 tok/s | 8 GiB separat belegt; Experten auf CPU und GPU |
| ca. 12 GB | 128K | 87.99 tok/s | 86.66 tok/s | 12 GiB separat belegt; Experten auf CPU und GPU |
| ca. 8 GB | 64K | 60.69 tok/s | 60.69 tok/s | FP8: kein Expert-Cache auf der GPU |

Dieselben drei Eingabetexte wurden je zweimal pro FP8-/INT8-Konfiguration mit den Sampling-Standardwerten des Modells verwendet und jeweils 128 Token einschließlich Thinking generiert. Die Geschwindigkeit ergibt sich aus der gesamten Tokenzahl geteilt durch die gesamte Decodierzeit, ohne Laden und Eingabeverarbeitung; Kontextwerte sind Cachekapazitäten. Die Zeilen mit weniger Speicher reservieren 8, 12 oder 16 GiB auf derselben 24-GB-GPU und verwenden die Kontextkapazität der jeweiligen VRAM-Klasse; sie sind keine Benchmarks physischer 16-/12-/8-GB-GPUs.

## Gemessene Geschwindigkeit — Gemma 4

Gemessen auf demselben Windows-PC mit 64 GB Arbeitsspeicher, Gemma 4 26B-A4B-it QAT Q4_0 und dem passenden QAT-MTP-Assistenten. Bei den Texteingaben über die API blieb der Bildencoder auf der GPU. MTP 3, CUDA Graphs sowie die modellspezifischen Thinking- und Sampling-Standardwerte waren aktiv; die Kontextkapazität wurde entsprechend der VRAM-Klasse festgelegt, Prompt-Chunks und Expertenplatzierung wurden automatisch gewählt.

Natives FP8 und INT8 wurden am **2026-10-21** gemessen.

| Verfügbarer VRAM | Kontextkapazität | INT8 (Standard) | Natives FP8 (experimentell) | Hinweise |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 113.20 tok/s | 112.31 tok/s | INT8 / FP8: Expertenberechnung auf der GPU |
| Ungefähr 16 GB | 256K | 100.08 tok/s | 96.67 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |
| Ungefähr 12 GB | 128K | 81.66 tok/s | 82.08 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |
| Ungefähr 8 GB | 64K | 51.54 tok/s | 53.41 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |

Dieselben drei Eingabetexte wurden je zweimal pro FP8-/INT8-Konfiguration mit den Sampling-Standardwerten des Modells verwendet und jeweils 128 Token einschließlich Thinking generiert. Die Geschwindigkeit ergibt sich aus der gesamten Tokenzahl geteilt durch die gesamte Decodierzeit, ohne Laden und Eingabeverarbeitung; Kontextwerte sind Cachekapazitäten. Die Zeilen mit weniger Speicher reservieren 8, 12 oder 16 GiB auf derselben 24-GB-GPU und verwenden die Kontextkapazität der jeweiligen VRAM-Klasse; sie sind keine Benchmarks physischer 16-/12-/8-GB-GPUs.

## Gemessene Geschwindigkeit — LFM2.5

Gemessen auf demselben Windows-PC mit 64 GB Arbeitsspeicher und dem offiziellen LFM2.5-24B-A2B Q4_K_M, ohne MTP oder Vision. CUDA Graphs sowie die modellspezifischen Thinking- und Sampling-Standardwerte waren aktiv; die Kontextkapazität wurde entsprechend der VRAM-Klasse festgelegt, Prompt-Chunks und Expertenplatzierung wurden automatisch gewählt.

Natives FP8 und INT8 wurden am **2026-10-21** gemessen.

| Verfügbarer VRAM | Kontextkapazität | INT8 | Natives FP8 (Standard) | Hinweise |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 128K | 140.58 tok/s | 141.80 tok/s | INT8 / FP8: Expertenberechnung auf der GPU |
| Ungefähr 16 GB | 128K | 135.63 tok/s | 138.21 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |
| Ungefähr 12 GB | 128K | 110.11 tok/s | 114.72 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |
| Ungefähr 8 GB | 64K | 71.10 tok/s | 73.92 tok/s | INT8 / FP8: Expertenberechnung auf GPU und CPU |

Dieselben drei Eingabetexte wurden je zweimal pro FP8-/INT8-Konfiguration mit den Sampling-Standardwerten des Modells verwendet und jeweils 128 Token einschließlich Thinking generiert. Die Geschwindigkeit ergibt sich aus der gesamten Tokenzahl geteilt durch die gesamte Decodierzeit, ohne Laden und Eingabeverarbeitung; Kontextwerte sind Cachekapazitäten. Die Zeilen mit weniger Speicher reservieren 8, 12 oder 16 GiB auf derselben 24-GB-GPU und verwenden die Kontextkapazität der jeweiligen VRAM-Klasse; sie sind keine Benchmarks physischer 16-/12-/8-GB-GPUs.

## Gemessene Geschwindigkeit — AMD Radeon RX 9070 XT (HIP, experimentell)

Gemessen am **2026-10-24** auf einem Windows-11-PC mit Radeon RX 9070 XT (gfx1201, 16 GB, steuert auch die Anzeige), 94 GB Arbeitsspeicher und HIP SDK 7.2, mit denselben Dateien Qwen3.6 UD-IQ3_XXS, Gemma 4 QAT Q4_0 (mit MTP-Assistent) und LFM2.5 Q4_K_M wie oben. Alles blieb über `run-api.bat` bei den Standardwerten: Das HIP-Backend fügt `--kv-whole` hinzu; Kontextkapazität, MTP und der GPU-Bildencoder wurden automatisch gewählt (LFM2.5 hat weder MTP noch Vision). Thinking war aktiv und das Sampling blieb bei den Modellstandards, nur jede Anfrage verwendete `seed` 1234.

| Modell | Kontextkapazität | Decodierung, erste 3 Anfragen nach dem Start (zwei Starts) | Decodierung, stabil (Median von 21 Anfragen) | Prompt mit etwa 2.7K Token, erste / spätere |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 256K | 68.3 / 71.4 tok/s | 70.9 tok/s (62.2 bis 78) | 514 und 485 / etwa 1200 tok/s |
| Gemma 4 QAT Q4_0 | 256K | 36.7 / 37.7 tok/s | 61.7 tok/s (50.6 bis 71.9) | 1518 und 1849 / etwa 1660 tok/s |
| LFM2.5 Q4_K_M | 128K | 80.9 / 86.2 tok/s | 101.1 tok/s (94.8 bis 110.6) | 2135 und 2214 / etwa 2410 tok/s |

Dieselben drei Eingabetexte wie oben wurden verwendet und jeweils 128 Token einschließlich Thinking generiert. Die Geschwindigkeit ergibt sich aus den generierten Token geteilt durch die Decodierzeit aus `/v1/status`, ohne Laden und Eingabeverarbeitung. „Erste 3 Anfragen“ sind die drei Texte in fester Reihenfolge direkt nachdem die API das Modell als geladen meldete, gemittelt, bei zwei getrennten Starts. „Stabil“ sind die Runden 4 bis 10 von zehn Runden der drei Texte bei einem Start (21 Anfragen), mit dem Bereich in Klammern. Die Prompt-Zeilen verwenden eine Eingabe von 2669 bis 2676 Token, jedes Mal mit anderer erster Zeile gesendet, damit nichts aus dem Cache kam: „erste“ ist die erste lange Anfrage bei jedem Start, „spätere“ die nächsten drei. Der VRAM-Spitzenwert des gesamten Adapters lag bei 15.55 bis 15.64 GiB von etwa 15.92 GiB, einschließlich etwa 2.4 GiB, die andere Programme belegten.

- Auf dieser Karte löst die Qwen-Engine mit dem wachsenden KV-Cache einen GPU-Fehler aus (siehe Issue #1); mit `--kv-whole` liefen alle drei Modelle ohne Fehler und ohne Windows-GPU-Watchdog-Dumps. Gemma 4 und LFM2.5 wurden nur in dieser Konfiguration ausgeführt.
- Qwen3.6 decodiert bei den ersten Anfragen im Rahmen der Streuung genauso schnell wie im stabilen Zustand, doch sein erster langer Prompt ist bei beiden Starts etwa 2.4-mal langsamer. Gemma 4 und LFM2.5 decodieren bei den ersten Anfragen langsamer (etwa 37 gegenüber 62 tok/s bzw. 81 bis 86 gegenüber 101 tok/s), bei beiden Starts. Die Engine-Logs zeigen starkes Auslagern im Expertencache während der ersten Anfrage (745 Swaps bei Gemma 4, später 164 bis 372) und eine schrumpfende VRAM-Größe über die ersten Anfragen, bei Gemma 4 von 8.5 auf 4.7 GiB und bei LFM2.5 von 11.6 auf 10.1 GiB. Ich habe nicht herausgefunden, warum er schrumpft oder ob das den langsamen Start erklärt.
- Bei festem Seed waren die Ausgaben der ersten drei Anfragen bei beiden Starts identisch. Derselbe Text später in einer Sitzung ergibt andere Ausgaben, die stabilen Zeilen mischen also Ausgabeunterschiede mit Geschwindigkeit.
- Jede Anfrage dauerte 1 bis 4 s, einzelne Anfragen schwanken um etwa 10 %: Unterschiede von wenigen tok/s nicht überinterpretieren. Es ist ein Rechner und eine Sitzung. Diese Karte hat 16 GB, die Tabellen oben verwenden eine 24-GB-Karte; die Zeilen sind daher nicht vergleichbar.

## Gemessene Geschwindigkeit — NVIDIA GeForce RTX 5060 (8 GB)

Gemessen am **2026-10-25** auf einem Windows-11-PC mit GeForce RTX 5060 (8 GB, Treiber 616.92), Core i7-10700K (8 Kerne), 32 GB DDR4-3200 und der GPU an einer PCIe-3.0-x8-Verbindung, mit denselben Dateien Qwen3.6 UD-IQ3_XXS, Gemma 4 QAT Q4_0 (mit MTP-Assistent) und LFM2.5 Q4_K_M wie oben. Alles blieb über `run-api.bat` bei den Standardwerten (CUDA-Backend; Kontextkapazität, KV-Typ, Eingabe-Chunk, Expertenplatzierung und MTP automatisch gewählt; LFM2.5 hat weder MTP noch Vision). Thinking war aktiv und die Sampling-Einstellungen blieben bei den Modellstandards, nur jede Anfrage verwendete `seed` 1234.

| Modell | Kontextkapazität | Decodierung, erste 3 Anfragen nach dem Start (zwei Starts) | Decodierung, stabil (Median von 21 Anfragen) | Prompt mit etwa 2.5K Token, erste / spätere |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 64K | 27.2 / 33.7 tok/s | 29.2 tok/s (24.1 bis 35.0) | 61 und 62 / etwa 104 (89 bis 118) tok/s |
| Gemma 4 QAT Q4_0 | 64K | 24.0 / 25.6 tok/s | 25.7 tok/s (22.2 bis 28.7) | 333 und 369 / etwa 364 (361 bis 371) tok/s |
| LFM2.5 Q4_K_M | 64K | 43.1 / 43.3 tok/s | 53.8 tok/s (42.4 bis 56.8) | 595 und 685 / etwa 716 (690 bis 762) tok/s |

Jede Anfrage generierte 128 Token einschließlich Thinking. Die Geschwindigkeit ergibt sich aus den generierten Token geteilt durch `predicted_ms` aus `/v1/status`, ohne Laden und Eingabeverarbeitung. „Erste 3 Anfragen“ sind drei Eingabetexte in fester Reihenfolge direkt nachdem die API das Modell als geladen meldete, bei zwei getrennten Starts (jeder Wert ist die Gesamtzahl der Token geteilt durch die Gesamtzeit der drei Anfragen). „Stabil“ sind die Runden 4 bis 10 von zehn Runden der drei Texte beim ersten Start (21 Anfragen), mit dem Bereich in Klammern. Die Prompt-Spalte verwendet je eine Eingabe von 2440 (Qwen3.6), 2446 (Gemma 4) bzw. 2545 (LFM2.5) Token, jedes Mal mit anderer erster Zeile gesendet, damit nichts gecacht wurde: „erste“ ist die erste lange Anfrage jedes Starts, „spätere“ die folgenden drei beider Starts. Die drei kurzen Eingabetexte und die lange Eingabe wurden für diese Messung geschrieben und sind nicht die der Abschnitte oben; die Zeilen sind daher nicht direkt mit ihnen vergleichbar. Der belegte GPU-Speicher nach den Läufen betrug 6.5 bis 6.9 GiB von etwa 7.9 GiB, einschließlich Desktop.

- Qwen3.6 hielt 12 (erster Start) bzw. 22 (zweiter Start) von 256 Experten pro Schicht im VRAM und die übrigen 11.4 GiB im angehefteten Arbeitsspeicher, bei einem Eingabe-Chunk von 512. Die beiden Starts hatten 4.6 und 5.3 GiB freien VRAM, und ihre späteren Prompt-Geschwindigkeiten lagen bei etwa 90 und 117 tok/s, was dazu passt, dass mehr residente Experten schneller sind; ich habe das nicht gesondert getestet.
- Die Prompt-Geschwindigkeiten sind im Vergleich zur 16-GB-Karte oben niedrig. Wahrscheinliche Gründe: Die meisten Experten liegen nicht im VRAM, die Experten außerhalb des VRAM werden auf 8 CPU-Threads berechnet (Qwen3.6) oder über eine PCIe-3.0-x8-Verbindung von grob 8 GB/s zur GPU gestreamt (Gemma 4 und LFM2.5, deren Logs gestagte Experten zeigen), und der freie VRAM ist zu klein, um einen größeren Chunk zu wählen. Andere `--chunk`-Werte wurden nicht ausprobiert; das ist also nicht bestätigt.
- Einzelne Anfragen schwanken um 10 % oder mehr, und die beiden Starts unterscheiden sich um bis zu 6 tok/s: kleine Unterschiede nicht überinterpretieren. Es ist ein Rechner und eine Sitzung.

## Welches Projekt sollte ich verwenden?

| Ihr Ziel und Ihre Hardware | Projekt |
| --- | --- |
| Sie möchten Qwen3.6 / Gemma 4 / LFM2.5-Inferenz ausschließlich per API unter **Windows**, mit **8–24 GB VRAM und höchstens 32 GB Arbeitsspeicher**, etwa auf einem PC mit einer gängigen Grafikkarte. | TinyStrata (**nur Windows**, in Entwicklung) |
| Sie verfügen sowohl über **eine GPU der RTX 3060-Klasse oder besser** als auch über **mindestens 64 GB Arbeitsspeicher**. | Wir empfehlen [Strata (Upstream)](https://github.com/Niko1221/Strata). |
| Sie möchten eine **integrierte Chat-Oberfläche**. | Wir empfehlen [Strata (Upstream)](https://github.com/Niko1221/Strata). |

## So wird der Speicher aufgeteilt — Qwen3.6

Ein MoE-Modell enthält viele **Experten**: kleinere neuronale Netze, die für jedes Token ausgewählt werden. TinyStrata hält so viele Experten wie möglich im GPU-Speicher (VRAM). Während der Generierung kann es ausgewählte, nicht zwischengespeicherte Experten auf der CPU unter Verwendung des Arbeitsspeichers berechnen. Bei der Verarbeitung größerer Eingabestapel kann es stattdessen Expertengewichte aus dem RAM durch temporäre GPU-Puffer streamen. Die vom Modell ausgewählten Experten bleiben Teil der Berechnung.

Die heruntergeladenen Modelldateien umfassen insgesamt etwa **15.0 GB**: **14.1 GB** für das Sprachmodell (einschließlich MTP-Gewichten) und **0.9 GB** für den Vision-Projektor (`mmproj-F16.gguf`). Das Sprachmodell muss nicht vollständig in den VRAM passen. Der GPU-Speicher benötigt außerdem Platz für den Vision-Encoder und Bildverarbeitungspuffer, den **KV-Cache** der Unterhaltung, die übrigen Gewichte des Modells und temporäre Berechnungspuffer:

![So teilt TinyStrata den Speicher zwischen GPU-VRAM und Arbeitsspeicher auf](assets/memory-sharing.png)

Dies ist ein Schema der Speicheraufteilung im standardmäßigen GPU-Bildmodus, keine Partitionierung mit festen Größen. Bei jedem Laden des Modells startet TinyStrata den GPU-Vision-Encoder und lässt ihn warmlaufen, bevor es das Sprachmodell lädt; seine Gewichte und aufgewärmten Arbeitspuffer bleiben im Speicher. Anschließend weist es die ersten KV-Seiten sowie Speicher für die übrigen Gewichte und Arbeitspuffer zu und bemisst den Experten-Cache nach dem verbleibenden freien VRAM. Unter Windows verwendet es den kleineren Wert aus dem freien CUDA-Speicher und dem vom NVIDIA-Treiber gemeldeten freien dedizierten VRAM, wenn beide verfügbar sind.

Der Cache-Inhalt ändert sich mit den verwendeten Experten. Beim standardmäßigen KV-Wachstum lassen kurze Unterhaltungen mehr VRAM für Experten; wächst die Sequenz, gibt der Expertencache Speicher an KV zurück. Neue kürzere Anfragen können ungenutzte KV-Seiten an den Expertencache zurückgeben. `run-api.bat --kv-whole` reserviert die gesamte KV-Kapazität vorab; `run-api.bat --kv-grow` stellt das Wachstum wieder her. Andere Anwendungen können weiterhin VRAM belegen.

Die Expertenplatzierung strebt normalerweise 768 MiB Reserve an. Im Qwen-Modus `--kv-whole` kann sie wie Strata bei knappem Platz gepackte Q8-Gewichtskopien freigeben, einen vorhandenen Attention-Score-Puffer verkleinern und die Reserve auf 256 MiB senken, damit alle Experten des Hauptmodells in den VRAM passen; MTP-Experten können auf der CPU laufen. `--vram-reserve-mib N` legt die Reserve ausdrücklich fest. Dies ist ein Ziel bei der Platzierung, keine dauerhafte Garantie für freien Speicher.

Mehr Experten im GPU-Speicher bedeuten normalerweise weniger CPU-Arbeit. Im Hybridmodus hält der Arbeitsspeicher auch Sicherungskopien der im VRAM zwischengespeicherten Experten vor. Zusätzlicher VRAM verringert den RAM-Verbrauch daher nicht Byte für Byte. Lassen Sie auch RAM für das Betriebssystem und andere Anwendungen frei. Mehr **RAM-Kapazität** ermöglicht es, Modelldaten vorzuhalten, doch die Geschwindigkeit hängt auch von CPU, RAM-Bandbreite und GPU ab. Der „gemeinsam genutzte GPU-Speicher“ von Windows unterscheidet sich davon, dass TinyStrata Experten gezielt auf der CPU berechnet.

Der KV-Cache bleibt im VRAM; für diesen Auslagerungsmechanismus hält der Arbeitsspeicher die Expertendaten vor. Eine lange Unterhaltung benötigt daher weiterhin genügend GPU-Speicher für ihren KV-Cache.

## Automatische Einstellungen — Qwen3.6

Neue Konfigurationen wählen KV/Attention und Kontextkapazität automatisch. Sie müssen kein Profil für 8 GB, 12 GB oder 16 GB auswählen.

Bei Qwen3.6 wählt eine RTX 3060 INT8. Unterstützte Umgebungen mit GPUs der RTX 40/50-Serie können natives FP8 wählen; die Engine prüft neben der Hardwaregeneration auch die kompilierten Kernel und die GPU-Fähigkeiten. Falls natives FP8 nicht verfügbar ist, verwendet der automatische Modus INT8. So wird ein unterstützter Pfad gewählt, ohne beim Start einen Geschwindigkeits- oder Qualitätsbenchmark auszuführen.

Die native Engine schätzt beim Start anhand des freien VRAM, der Kontextkapazität und des Platzbedarfs der GPU-Experten eine Blockgröße von 512, 1024, 2048 oder 4096 Token. Größere Blöcke können wiederholte Gewichtstransfers vom RAM zur GPU während der Eingabeverarbeitung reduzieren. `run-api.bat --chunk N` legt die Größe fest; `run-api.bat --chunk auto` stellt die native automatische Auswahl wieder her.

Die Kontextkapazität ist die logische Obergrenze für Eingabe und generierte Ausgabe zusammen. Standardmäßig wird physischer KV-Speicher mit wachsender Unterhaltung zugewiesen, statt bei einer kurzen Unterhaltung die gesamte Kapazität vorab zu belegen. Lange Dokumente erfordern weiterhin mehr Berechnungen und schließlich mehr KV-Speicher.

Bei voller Kapazität benötigt Qwens INT8-KV einschließlich der MTP-Schicht etwa **0.71 GiB bei 64K, 1.42 GiB bei 128K und 2.84 GiB bei 256K**; weitere Gewichte und Arbeitspuffer kommen hinzu. Natives FP8 benötigt in dieser Implementierung ungefähr ebenso viel KV-Speicher wie INT8; sein Vorteil liegt in einem anderen Berechnungspfad und nicht in einem wesentlich kleineren Cache.

Die Kontextkapazität wird anhand des gesamten GPU-VRAM gewählt; der Expertencache wird anschließend in den beim Laden verbleibenden freien VRAM eingepasst. Andere Anwendungen mit VRAM-Bedarf lassen weniger Platz für GPU-gecachte Experten. Die API weist Eingaben zurück, die die Kontextkapazität überschreiten, anstatt sie stillschweigend zu kürzen.

Die Quantisierung der GGUF-Gewichte und die KV-Präzision sind voneinander unabhängig. Das ausgewählte UD-IQ3_XXS-Modell hat komprimierte Gewichte; INT8 oder FP8 beschreibt den Unterhaltungscache und seinen Attention-Pfad. KV mit geringerer Präzision kann Antworten verändern, daher versprechen diese Modi keine mit FP32-KV identischen Ausgaben. Die Implementierung behält alle sichtbaren Kontextpositionen und jeden vom Modell ausgewählten Experten bei.

Neue Launcher-Konfigurationen verwenden MTP 3, das mehrere Token im Voraus vorschlägt und sie mit dem Hauptmodell überprüft. Das kann die Generierung beschleunigen, doch der Gewinn hängt davon ab, wie viele Vorschläge akzeptiert werden. Die Überprüfung soll das Dekodierverhalten des Hauptmodells bewahren; Unterschiede in numerischen Ausführungspfaden können dennoch Token verändern. Deshalb verspricht aktiviertes MTP keine mit deaktiviertem MTP identischen Ausgaben.

MTP unterstützt sowohl Greedy-Generierung als auch temperaturbasiertes Sampling, einschließlich Wiederholungs-, Häufigkeits- und Präsenzstrafen. Bei Sampling-Anfragen zieht das Hauptmodell für jede Überprüfungszeile eine Stichprobe, und nur das übereinstimmende Präfix des Entwurfs wird akzeptiert.

Explizite Einstellungen überschreiben die automatische Auswahl, und bereits manuell gewählte Einstellungen bleiben wirksam. Beispielsweise legt `run-api.bat --kv-type int8` INT8 fest; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` stellt die automatische Auswahl von KV, Kontext und Expertenplatzierung wieder her.

## API-Server

**TinyStrata bietet ausschließlich eine API.** Es stellt OpenAI-kompatible APIs für Chat Completions (`/v1/chat/completions`) und Responses (`/v1/responses`) sowie eine Anthropic-kompatible Messages-API (`/v1/messages`) bereit, einschließlich gestreamter Generierung. Verbinden Sie einen API-Client oder ein externes Frontend; TinyStrata enthält keine integrierte Chat-Oberfläche.

Neue Konfigurationen verwenden modellspezifische Sampling-Standardwerte. Werte pro Anfrage haben Vorrang; gespeicherte benutzerdefinierte Einstellungen bleiben erhalten.

| Einstellung | Qwen3.6 | Gemma 4 | LFM2.5 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.7 | 1.0 | 0.1 |
| `top_p` | 0.95 | 0.95 | 1.0 |
| `top_k` | 20 | 64 | 50 |
| `min_p` | 0.0 | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.05 |
| `enable_thinking` | true | true | true (immer aktiv) |
| `max_tokens` (einschließlich Thinking) | 32768 | 32768 | 32768 |

Die API lauscht standardmäßig unter `http://127.0.0.1:18080`. Um einen anderen Port zu verwenden, führen Sie `run-api.bat --port 18081` im installierten TinyStrata-Ordner aus.

Bei Qwen verarbeitet `run-api.bat --vision gpu` Bilder auf der GPU (Standard für neue Konfigurationen); `--vision cpu` nutzt die CPU und `--vision no` deaktiviert Bilder. `--vision-tokens N` setzt die maximale Zahl der Bild-Token (Standard: 4096). Diese Auswahl wird gespeichert und bei späteren Starts ohne diese Optionen beibehalten.

Die API kann starten, bevor die Modelldateien installiert sind. Wenn Qwen ausgewählt ist, lädt `POST /v1/model/download` mit einem leeren JSON-Body (`{}`) das festgelegte [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS-Sprachmodell](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, einschließlich MTP-Gewichten) und seinen [Vision-Projektor](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB) herunter, insgesamt etwa **15.0 GB**. Die SHA-256-Prüfsummen beider Dateien werden geprüft, der Tokenizer wird extrahiert und die Dateien werden geladen. `GET /v1/model/download` meldet den Fortschritt; unterbrochene Downloads können fortgesetzt werden. Eine wiederholte Download-Anfrage verwendet den aktiven Auftrag weiter, und `POST /v1/model/download/cancel` kann die Übertragung oder Überprüfung abbrechen. Ein abgeschlossener Download bedeutet noch keine Inferenzbereitschaft: Warten Sie, bis der Download-Status `ready` lautet und `/health` `loaded: true` sowie bei aktivierten Bildern auch `images: true` meldet.

Das Installationsprogramm bereitet die Engine-Programmdateien und GPU-Laufzeitbibliotheken vor und lädt standardmäßig Qwen und seinen Vision-Projektor herunter. Bestehende Konfigurationen behalten ihr ausgewähltes lokales Modell bei, bis Sie es ausdrücklich ändern.

## Danksagungen

TinyStrata ist ein unabhängiger Fork von [Strata](https://github.com/Niko1221/Strata) und basiert in weiten Teilen auf dessen Implementierung. Vielen Dank an [Niko](https://github.com/Niko1221) und die Strata-Community für die Engine und die Arbeit, die TinyStrata möglich gemacht haben.

Vielen Dank auch an [Aaron Flin](https://github.com/aflin) und das [MoMoE-Projekt](https://github.com/aflin/momoe) für den in TinyStrata angepassten Code für KV-Cache, Attention und quantisierte Experten.

## Lizenz

MIT. Siehe [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Upstream-Attribution: [NOTICE](NOTICE). GGML: [MIT-Lizenz](third_party/ggml/LICENSE).
