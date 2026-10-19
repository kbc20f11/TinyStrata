# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata ist eine schnelle Inferenz-Engine für MoE-Modelle der 30B-Klasse, die ausschließlich für Windows und die Nutzung per API entwickelt wird.** Sie richtet sich vor allem an PCs, die mit gängigen Grafikkarten mit **8–24 GB VRAM** ausgestattet sind und über **höchstens 32 GB Arbeitsspeicher** verfügen.

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

**Qwen3.6-35B-A3B UD-IQ3_XXS** verwendet den festgelegten [Unsloth-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) und den Vision-Projektor `mmproj-F16.gguf`. Die beiden Dateien umfassen zusammen etwa **15.0 GB**: Sprachmodell einschließlich MTP-Gewichten 14.1 GB und Projektor 0.9 GB. Unterstützt werden ein GPU-residenter Bildencoder, ein automatisch an den freien VRAM angepasster Expertencache, CPU-Expertenberechnung aus dem Arbeitsspeicher, Gewichtsstreaming bei der Eingabeverarbeitung, eine automatisch gewählte Kontextkapazität von bis zu **256K** und **MTP 3**. Ausgewählte Experten werden nicht ausgelassen. Der KV-Cache unterstützt **FP32, INT8 und FP8**; die automatische Auswahl verwendet natives FP8, wenn GPU und kompilierte Kernel dies unterstützen, andernfalls INT8. MTP funktioniert sowohl mit Greedy-Decodierung als auch mit temperaturbasiertem Sampling.

**Gemma 4 26B-A4B-it QAT Q4_0** verwendet [Googles offizielles Zielmodell und den Vision-Projektor](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) zusammen mit einem [passenden Unsloth-QAT-Q4_0-MTP-Assistenten](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Die drei Downloads umfassen **15.886 GB**: Zielmodell 14.439 GB, Projektor 1.195 GB und Assistent 0.252 GB. Unterstützt werden ein GPU-residenter Bildencoder, ein adaptiver GPU-Expertencache, CPU-Expertenberechnung aus dem Arbeitsspeicher und MTP 3 über dieselbe API. Ausgewählte Experten werden nicht ausgelassen. Gemma unterstützt **FP16- und INT8-KV**; die automatische Auswahl verwendet derzeit **FP16**. Bildeingabe, MTP 3, RAM-Mitnutzung und Modellwechsel wurden mit dem realen Modell getestet.

LFM verwendet die offizielle [LiquidAI Q4_K_M GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (**14.53 GB**), **FP16-KV** und bis zu **128K Kontext**, ohne MTP oder Vision. Automatischer VRAM-Expertencache, CPU-Expertenberechnung, Gewichtsstreaming bei der Eingabe und CUDA Graphs werden genutzt; ausgewählte Experten werden nicht ausgelassen. Die Standardwerte sind `temperature=0.1`, `top_k=50` und `repetition_penalty=1.05`; Reasoning ist immer aktiv. Die Einstellungen für Qwen, Gemma und LFM werden getrennt gespeichert. Die LFM-Engine wird noch validiert und garantiert keine identischen Tokenfolgen zu llama.cpp oder bei unterschiedlicher CPU/GPU-Aufteilung.

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

Die folgenden Messungen wurden am 2026-10-19 unter Windows mit einer RTX PRO 4000 Blackwell SFF Edition (24 GB), einem Ryzen 9 7945HX3D und dem festgelegten Unsloth MTP UD-IQ3_XXS-Modell durchgeführt. Modell und GPU-Vision-Projektor waren geladen und aufgewärmt. Verwendet wurden die API-Standardeinstellungen: Thinking aktiviert, Sampling mit Temperatur 0.7, MTP 3, Prompt-Blöcke mit 512 Token sowie automatische Expertenplatzierung und Kontextauswahl. Der optionale hybride CUDA-Graph-Pfad war standardmäßig ausgeschaltet; die normalen CUDA Graphs bei vollständigem GPU-Betrieb blieben aktiv. Auf dieser GPU wählte `auto` natives FP8; für die INT8-Spalte wurde nur der KV-Typ erzwungen.

**Messungen auf realen Systemen mit 8-GB-, 12-GB- und 16-GB-GPUs sind geplant.** Die Zeilen mit reduziertem Speicher in der folgenden Tabelle wurden auf derselben 24-GB-GPU mit begrenztem verfügbarem Speicher gemessen und sind keine Benchmarks physischer GPUs mit 8/12/16 GB.

| Verfügbarer Speicher | Kontextkapazität | INT8-Generierung | Native FP8-Generierung | Anmerkungen |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | Alle Gewichte im VRAM |
| ca. 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | 8 GiB separat belegt; Experten auf CPU und GPU |
| ca. 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | 12 GiB separat belegt; Experten auf CPU und GPU |
| ca. 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | 16 GiB separat belegt; Experten auf CPU und GPU |

## Gemessene Geschwindigkeit — Gemma 4

Gemessen am 2026-10-19 auf demselben Windows-PC mit RTX PRO 4000 Blackwell SFF Edition (24 GB), Ryzen 9 7945HX3D und 64 GB Arbeitsspeicher. Verwendet wurden Googles Gemma 4 26B-A4B-it QAT Q4_0 und der passende Unsloth-QAT-MTP-Assistent, ein GPU-residenter Bildencoder, Texteingaben über die API, MTP 3 und aktivierte CUDA Graphs. Es gelten die Google-Standardwerte: Thinking aktiv, `temperature=1.0`, `top_p=0.95`, `top_k=64`. Kontext und Expertenplatzierung wurden automatisch gewählt; die automatischen Prompt-Chunks umfassten 1120 Token.

| Verfügbarer VRAM | Kontextkapazität | FP16-Generierung | INT8-Generierung | Hinweise |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 105.56 tok/s | 118.62 tok/s | FP16: Expertenberechnung auf GPU und CPU; INT8: alle Experten im GPU-Speicher |
| Etwa 16 GB | 128K | 88.46 tok/s | 86.67 tok/s | Expertenberechnung auf GPU und CPU |
| Etwa 12 GB | 64K | 68.16 tok/s | 76.47 tok/s | Expertenberechnung auf GPU und CPU |
| Etwa 8 GB | 64K | 48.22 tok/s | 48.97 tok/s | Expertenberechnung während der Generierung auf der CPU |

Drei feste Eingabetexte wurden jeweils einmal mit 128 Ausgabetoken einschließlich Thinking ausgeführt. Die Geschwindigkeit ist die gesamte Tokenzahl geteilt durch die gesamte Decodierzeit, ohne Laden und Eingabeverarbeitung. Für die Bedingungen mit weniger verfügbarem Speicher wurden separat 8, 12 bzw. 16 GiB auf derselben 24-GB-GPU reserviert; es sind keine Benchmarks physischer 16-, 12- oder 8-GB-GPUs. 64K/128K/256K bezeichnet die Cache-Kapazität, nicht die Eingabelänge.

## Gemessene Geschwindigkeit — LFM2.5 (experimentell)

Gemessen auf demselben Windows-PC mit 64 GB Arbeitsspeicher und dem offiziellen LFM2.5-24B-A2B Q4_K_M, FP16-KV, ohne MTP oder Vision, mit aktivierten CUDA Graphs. Es gelten die aktuellen API-Standardwerte: Thinking aktiv, `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`. Kontextkapazität, Prompt-Chunks (jeweils 256 Token) und Expertenverteilung wurden automatisch gewählt.

| Verfügbarer VRAM | Kontextkapazität | FP16-Generierung | Hinweise |
| --- | ---: | ---: | --- |
| 24 GB | 128K | 151.71 tok/s | Alle Gewichte im VRAM |
| Etwa 16 GB | 128K | 111.71 tok/s | Expertenberechnung auf GPU und CPU |
| Etwa 12 GB | 128K | 70.80 tok/s | Expertenberechnung auf GPU und CPU |
| Etwa 8 GB | 64K | 51.20 tok/s | Expertenberechnung auf GPU und CPU |

Die gleichen drei Eingabetexte wie beim Qwen-Benchmark wurden jeweils zweimal ausgeführt, mit je 128 generierten Token einschließlich Thinking. Die Geschwindigkeit ist die gesamte Tokenzahl geteilt durch die gesamte Decodierzeit, ohne Laden und Prompt-Verarbeitung. Die Zeilen mit weniger Speicher begrenzen den verfügbaren VRAM derselben 24-GB-GPU und sind keine Benchmarks physischer 8-/12-/16-GB-GPUs. Modell und Sampling-Einstellungen unterscheiden sich von der Qwen-Tabelle.

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

Dies ist ein Schema der Speicheraufteilung im standardmäßigen GPU-Bildmodus, keine Partitionierung mit festen Größen. Bei jedem Laden des Modells startet TinyStrata den GPU-Vision-Encoder und lässt ihn warmlaufen, bevor es das Sprachmodell lädt; seine Gewichte und aufgewärmten Arbeitspuffer bleiben im Speicher. Anschließend weist es Speicher für den KV-Cache, die übrigen Gewichte und Arbeitspuffer zu und bemisst den Experten-Cache nach dem verbleibenden freien VRAM. Unter Windows verwendet es den kleineren Wert aus dem freien CUDA-Speicher und dem vom NVIDIA-Treiber gemeldeten freien dedizierten VRAM, wenn beide verfügbar sind.

Der Cache-Inhalt ändert sich, wenn andere Experten verwendet werden; die Cache-Kapazität wird beim erneuten Laden des Modells neu berechnet. Später gestartete Anwendungen können weiterhin VRAM belegen.

Die Expertenplatzierung strebt normalerweise 768 MiB Reserve an. Wie Strata kann sie bei knappem Platz gepackte Q8-Gewichtskopien freigeben, einen vorhandenen Attention-Score-Puffer verkleinern und die Reserve auf 256 MiB senken, damit alle Experten des Hauptmodells in den VRAM passen; MTP-Experten können auf der CPU laufen. `--vram-reserve-mib N` legt die Reserve ausdrücklich fest. Dies ist ein Ziel bei der Platzierung, keine dauerhafte Garantie für freien Speicher.

Mehr Experten im GPU-Speicher bedeuten normalerweise weniger CPU-Arbeit. Im Hybridmodus hält der Arbeitsspeicher auch Sicherungskopien der im VRAM zwischengespeicherten Experten vor. Zusätzlicher VRAM verringert den RAM-Verbrauch daher nicht Byte für Byte. Lassen Sie auch RAM für das Betriebssystem und andere Anwendungen frei. Mehr **RAM-Kapazität** ermöglicht es, Modelldaten vorzuhalten, doch die Geschwindigkeit hängt auch von CPU, RAM-Bandbreite und GPU ab. Der „gemeinsam genutzte GPU-Speicher“ von Windows unterscheidet sich davon, dass TinyStrata Experten gezielt auf der CPU berechnet.

Der KV-Cache bleibt im VRAM; für diesen Auslagerungsmechanismus hält der Arbeitsspeicher die Expertendaten vor. Eine lange Unterhaltung benötigt daher weiterhin genügend GPU-Speicher für ihren KV-Cache.

## Automatische Einstellungen — Qwen3.6

Neue Konfigurationen wählen KV/Attention und Kontextkapazität automatisch. Sie müssen kein Profil für 8 GB, 12 GB oder 16 GB auswählen.

Eine RTX 3060 wählt INT8. Unterstützte Umgebungen mit GPUs der RTX 40/50-Serie können natives FP8 wählen; die Engine prüft neben der Hardwaregeneration auch die kompilierten Kernel und die GPU-Fähigkeiten. Falls natives FP8 nicht verfügbar ist, verwendet der automatische Modus INT8. So wird ein unterstützter Pfad gewählt, ohne beim Start einen Geschwindigkeits- oder Qualitätsbenchmark auszuführen. **Native FP8-Attention bleibt experimentell.**

Die native Engine schätzt beim Start anhand des freien VRAM, der Kontextkapazität und des Platzbedarfs der GPU-Experten eine Blockgröße von 512, 1024, 2048 oder 4096 Token. Größere Blöcke können wiederholte Gewichtstransfers vom RAM zur GPU während der Eingabeverarbeitung reduzieren. `run-api.bat --chunk N` legt die Größe fest; `run-api.bat --chunk auto` stellt die native automatische Auswahl wieder her.

Die Kontextkapazität ist der maximale Speicherplatz, der für Eingabe und generierte Ausgabe zusammen reserviert wird. Eine höhere Kapazität benötigt selbst bei einer kurzen Unterhaltung mehr KV-Speicher und lässt weniger VRAM für Experten übrig. Eine bestimmte Kapazität bedeutet auch nicht bei jeder Unterhaltungslänge dieselbe Geschwindigkeit: Die tatsächliche Verarbeitung eines langen Dokuments erfordert mehr Arbeit. Zur Einordnung: INT8-KV allein benötigt einschließlich der MTP-Schicht etwa **0.71 GiB bei 64K, 1.42 GiB bei 128K und 2.84 GiB bei 256K**; weitere Gewichte und Arbeitspuffer kommen hinzu. Natives FP8 benötigt in dieser Implementierung ungefähr ebenso viel KV-Speicher wie INT8; sein Vorteil liegt in einem anderen Berechnungspfad und nicht in einem wesentlich kleineren Cache.

Freier VRAM bezeichnet den zum Ladezeitpunkt verfügbaren Speicher, nicht die Zahl auf der GPU-Verpackung. Andere Anwendungen können daher dazu führen, dass ein kleinerer Kontext ausgewählt wird. Falls sich die Kapazität beim erneuten Laden verringert, weist die API eine nicht mehr passende Eingabe zurück, anstatt sie stillschweigend zu kürzen. Unter Windows prüft die automatische Kontextauswahl sowohl den freien CUDA-Speicher als auch die geräteweite Angabe des NVIDIA-Treibers und verwendet den kleineren Wert, wenn beide Abfragen erfolgreich sind.

Die Quantisierung der GGUF-Gewichte und die KV-Präzision sind voneinander unabhängig. Das ausgewählte UD-IQ3_XXS-Modell hat komprimierte Gewichte; INT8 oder FP8 beschreibt den Unterhaltungscache und seinen Attention-Pfad. KV mit geringerer Präzision kann Antworten verändern, daher versprechen diese Modi keine mit FP32-KV identischen Ausgaben. Die Implementierung behält alle sichtbaren Kontextpositionen und jeden vom Modell ausgewählten Experten bei.

Neue Launcher-Konfigurationen verwenden MTP 3, das mehrere Token im Voraus vorschlägt und sie mit dem Hauptmodell überprüft. Das kann die Generierung beschleunigen, doch der Gewinn hängt davon ab, wie viele Vorschläge akzeptiert werden. Die Überprüfung soll das Dekodierverhalten des Hauptmodells bewahren; Unterschiede in numerischen Ausführungspfaden können dennoch Token verändern. Deshalb verspricht aktiviertes MTP keine mit deaktiviertem MTP identischen Ausgaben.

MTP unterstützt sowohl Greedy-Generierung als auch temperaturbasiertes Sampling, einschließlich Wiederholungs-, Häufigkeits- und Präsenzstrafen. Bei Sampling-Anfragen zieht das Hauptmodell für jede Überprüfungszeile eine Stichprobe, und nur das übereinstimmende Präfix des Entwurfs wird akzeptiert.

Explizite Einstellungen überschreiben die automatische Auswahl, und bereits manuell gewählte Einstellungen bleiben wirksam. Beispielsweise legt `run-api.bat --kv-type int8` INT8 fest; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` stellt die automatische Auswahl von KV, Kontext und Expertenplatzierung wieder her.

## API-Server

**TinyStrata bietet ausschließlich eine API.** Es stellt OpenAI-kompatible APIs für Chat Completions (`/v1/chat/completions`) und Responses (`/v1/responses`) sowie eine Anthropic-kompatible Messages-API (`/v1/messages`) bereit, einschließlich gestreamter Generierung. Verbinden Sie einen API-Client oder ein externes Frontend; TinyStrata enthält keine integrierte Chat-Oberfläche.

Bei Qwen ist Thinking standardmäßig aktiviert, mit `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` und maximal **32,768 Ausgabetoken** einschließlich Thinking. Einstellungen pro Anfrage haben Vorrang.

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
