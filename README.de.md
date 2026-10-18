# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**Lokale MoE-Inferenz ausschließlich unter Windows und per API, mit Schwerpunkt auf PCs mit 8–24 GB VRAM und bis zu 32 GB Arbeitsspeicher.** Zur Zielgruppe gehören gewöhnliche Gaming-PCs und Gaming-Laptops.

## Warum TinyStrata?

Angesichts steigender Preise für PC-Komponenten wird ein leistungsstarker „1337 PC“ für viele Menschen immer unerschwinglicher.

TinyStrata wurde entwickelt, um KI lokal mit hoher Geschwindigkeit auf gewöhnlichen Gaming-PCs und Gaming-Laptops mit einer GPU der RTX 3060-Klasse oder besser auszuführen. Unser Schwerpunkt liegt darauf, kleinere MoE-Modelle so schnell wie möglich auszuführen.

Unsere Hauptzielgruppe sind PCs mit **8–24 GB VRAM und höchstens 32 GB Arbeitsspeicher**. Wir glauben, dass dieser Bereich die gängigsten Konfigurationen von Gaming-PCs und Gaming-Laptops abdeckt.

Auf Basis von Strata soll TinyStrata den VRAM-Verbrauch senken und die Inferenz für Qwen3.6-35B-A3B beschleunigen. Wir möchten begrenzten GPU-Speicher effizient nutzen, damit Sie auf Ihrem eigenen PC mit langen Unterhaltungen und Dokumenten arbeiten können.

Wir pflegen diesen Fork, um die Entwicklung auf schnelle lokale Inferenz unter Windows und deren Bereitstellung über eine API zu konzentrieren. Dieser klar umrissene Umfang bestimmt unsere Implementierung und Tests.

Außerdem möchten wir den Einstieg so einfach wie möglich machen. TinyStrata wählt Inferenz- und Speichereinstellungen automatisch passend zu Ihrer Hardware, sodass Sie die API ohne komplizierte Konfiguration oder manuelle Feinabstimmung nutzen können.

## Warum keine integrierte Benutzeroberfläche?

Sobald TinyStrata läuft, können Sie Ihre bevorzugte API-kompatible Benutzeroberfläche oder Agenten-Laufzeitumgebung verbinden. Wir konzentrieren uns auf das Inferenz-Backend, damit Sie die Oberfläche und den Arbeitsablauf wählen können, die zu Ihnen passen. Falls Ihnen die Einrichtung der Verbindung zu aufwendig erscheint, können Sie einen KI-Agenten damit beauftragen.

## Modellschwerpunkt

TinyStrata konzentriert sich derzeit ausschließlich auf [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Wir werden die Unterstützung eines weiteren Modells erwägen, wenn ein MoE-Modell für schnelle lokale Inferenz auf gewöhnlichen Gaming-PCs und Gaming-Laptops besser geeignet ist als Qwen3.6-35B-A3B.

## Schnellinstallation

Führen Sie diesen Befehl in 64-Bit-PowerShell aus, um TinyStrata zu installieren und die API unter `http://127.0.0.1:18080/v1` zu starten:

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Um TinyStrata nach einem Windows-Neustart erneut zu starten, führen Sie diesen Befehl aus oder doppelklicken Sie auf dieselbe `run-api.bat`:

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

## Gemessene Geschwindigkeit

Die folgenden Messungen wurden am 2026-10-17 unter Windows mit einer RTX PRO 4000 Blackwell SFF Edition (24 GB), einem Ryzen 9 7945HX3D und dem festgelegten Unsloth MTP UD-IQ3_XXS-Modell durchgeführt. Verwendet wurden Greedy-Generierung, MTP 3, die unten angegebenen Kontextkapazitäten, Prompt-Blöcke mit 512 Token, automatische Expertenplatzierung und der aktivierte optionale hybride CUDA-Graph-Pfad.

| Verfügbarer Speicher | Kontextkapazität | INT8-Generierung | Native FP8-Generierung |
| --- | ---: | ---: | ---: |
| Gesamte 24-GB-GPU; alle Experten im VRAM | 256K | 121.28 tok/s | 122.73 tok/s |
| 8 GiB separat belegt; etwa 16 GB verbleibend | 256K | 96.98 tok/s | 112.39 tok/s |
| 12 GiB separat belegt; etwa 12 GB verbleibend | 128K | 104.28 tok/s | 107.61 tok/s |
| 16 GiB separat belegt; etwa 8 GB verbleibend | 64K | 81.43 tok/s | 86.69 tok/s |

Dies sind zusammengefasste Dekodiergeschwindigkeiten für drei kurze Eingaben (34, 46 und 1,599 Token), die jeweils zweimal 128 Token erzeugten. Laden und Eingabeverarbeitung sind nicht enthalten. **64K–256K sind zugewiesene Cache-Kapazitäten, keine Eingabelängen.** Alle Zeilen mit reduziertem Speicher nutzten unabhängige Speicherzuweisungen auf derselben 24-GB-GPU und führten tatsächlich Expertenberechnungen auf der CPU aus; es handelt sich nicht um Benchmarks physischer GPUs mit 16/12/8 GB.

Die Werte für 24/16 GB stammen aus dem früheren Lauf. Die ergänzten Fälle mit 12/8 GB nutzten den aktuellen Build und die automatische Kontextauswahl (128K/64K). Auch die Kontextkapazität beeinflusst die Geschwindigkeit, daher zeigen diese Zeilen unterschiedliche praxisnahe Einstellungen. Der Vorteil von FP8 ist hier ein Messergebnis unter diesen Bedingungen, keine allgemeine Garantie und kein Qualitätsvergleich.

## Welches Projekt sollte ich verwenden?

| Ihr Ziel und Ihre Hardware | Projekt |
| --- | --- |
| Sie möchten Qwen3.6-35B-A3B-Inferenz ausschließlich per API unter **Windows**, mit **8–24 GB VRAM und höchstens 32 GB Arbeitsspeicher**, etwa auf einem gewöhnlichen Gaming-PC oder einem Gaming-Laptop. | TinyStrata (**nur Windows**, in Entwicklung) |
| Sie verfügen sowohl über **eine GPU der RTX 3060-Klasse oder besser** als auch über **mindestens 64 GB Arbeitsspeicher**. | Wir empfehlen [Strata (Upstream)](https://github.com/Niko1221/Strata). |
| Sie möchten eine **integrierte Chat-Oberfläche**. | Wir empfehlen [Strata (Upstream)](https://github.com/Niko1221/Strata). |

**TinyStrata unterstützt ausschließlich Windows.** Linux und macOS liegen außerhalb des Unterstützungsumfangs dieses Projekts. Diese Angaben zum Arbeitsspeicher dienen als Richtwerte für die Projektwahl. Der tatsächliche Speicherbedarf hängt vom Modell, der Quantisierung, der Kontextlänge und dem verfügbaren VRAM ab.

MoE-Inferenz mit GPU-Speicher und Arbeitsspeicher kann das Zwischenspeichern von Experten im VRAM und die Berechnung einiger Experten auf der CPU umfassen. Angaben zu unterstützten Modellen, Betriebssystemen und GPUs finden Sie in der README des jeweiligen Projekts.

## So wird der Speicher aufgeteilt

Ein MoE-Modell enthält viele **Experten**: kleinere neuronale Netze, die für jedes Token ausgewählt werden. TinyStrata hält so viele Experten wie möglich im GPU-Speicher (VRAM). Während der Generierung kann es ausgewählte, nicht zwischengespeicherte Experten auf der CPU unter Verwendung des Arbeitsspeichers berechnen. Bei der Verarbeitung größerer Eingabestapel kann es stattdessen Expertengewichte aus dem RAM durch temporäre GPU-Puffer streamen. Die vom Modell ausgewählten Experten bleiben Teil der Berechnung.

Die heruntergeladenen Modelldateien umfassen insgesamt etwa **15.0 GB**: **14.1 GB** für das Sprachmodell (einschließlich MTP-Gewichten) und **0.9 GB** für den Vision-Projektor (`mmproj-F16.gguf`). Das Sprachmodell muss nicht vollständig in den VRAM passen. Der GPU-Speicher benötigt außerdem Platz für den Vision-Encoder und Bildverarbeitungspuffer, den **KV-Cache** der Unterhaltung, die übrigen Gewichte des Modells und temporäre Berechnungspuffer:

![So teilt TinyStrata den Speicher zwischen GPU-VRAM und Arbeitsspeicher auf](assets/memory-sharing.png)

Dies ist ein Schema der Speicheraufteilung, keine Partitionierung mit festen Größen. Bei jedem Laden des Modells startet TinyStrata den GPU-Vision-Encoder und lässt ihn warmlaufen, bevor es das Sprachmodell lädt. Anschließend weist es Speicher für den KV-Cache, die übrigen Gewichte und Arbeitspuffer zu und passt den Experten-Cache unter Wahrung einer Reserve in den verbleibenden freien VRAM ein. Unter Windows prüft es sowohl die CUDA-Angabe zum freien Speicher als auch die geräteweite Angabe des NVIDIA-Treibers und verwendet den kleineren verfügbaren Wert, wenn beide Abfragen erfolgreich sind. Falls die Speicherzuweisung fehlschlägt oder zu wenig Reserve lässt, versucht es es mit weniger Experten erneut.

Der Cache-Inhalt ändert sich, wenn andere Experten verwendet werden; die Cache-Kapazität wird beim erneuten Laden des Modells neu berechnet. Später gestartete Anwendungen können weiterhin VRAM belegen. Die Reserve ist ein Spielraum bei der Expertenplatzierung, keine dauerhafte Garantie, dass 768 MiB frei bleiben.

Mehr Experten im GPU-Speicher bedeuten normalerweise weniger CPU-Arbeit. Im Hybridmodus hält der Arbeitsspeicher auch Sicherungskopien der im VRAM zwischengespeicherten Experten vor. Zusätzlicher VRAM verringert den RAM-Verbrauch daher nicht Byte für Byte. Lassen Sie auch RAM für das Betriebssystem und andere Anwendungen frei. Mehr **RAM-Kapazität** ermöglicht es, Modelldaten vorzuhalten, doch die Geschwindigkeit hängt auch von CPU, RAM-Bandbreite und GPU ab. Der „gemeinsam genutzte GPU-Speicher“ von Windows unterscheidet sich davon, dass TinyStrata Experten gezielt auf der CPU berechnet.

Der KV-Cache bleibt im VRAM; für diesen Auslagerungsmechanismus hält der Arbeitsspeicher die Expertendaten vor. Eine lange Unterhaltung benötigt daher weiterhin genügend GPU-Speicher für ihren KV-Cache.

## Automatische Einstellungen

Neue Konfigurationen wählen KV/Attention und Kontextkapazität automatisch. Sie müssen kein Profil für 8 GB, 12 GB oder 16 GB auswählen.

![Automatische Hardwareeinstellungen für Attention, KV-Cache und Kontextkapazität](assets/automatic-settings.png)

Eine RTX 3060 wählt INT8. Unterstützte Umgebungen mit GPUs der RTX 40/50-Serie können natives FP8 wählen; die Engine prüft neben der Hardwaregeneration auch die kompilierten Kernel und die GPU-Fähigkeiten. Falls natives FP8 nicht verfügbar ist, verwendet der automatische Modus INT8. So wird ein unterstützter Pfad gewählt, ohne beim Start einen Geschwindigkeits- oder Qualitätsbenchmark auszuführen. **Native FP8-Attention bleibt experimentell.**

Die Kontextkapazität ist der maximale Speicherplatz, der für Eingabe und generierte Ausgabe zusammen reserviert wird. Eine höhere Kapazität benötigt selbst bei einer kurzen Unterhaltung mehr KV-Speicher und lässt weniger VRAM für Experten übrig. Eine bestimmte Kapazität bedeutet auch nicht bei jeder Unterhaltungslänge dieselbe Geschwindigkeit: Die tatsächliche Verarbeitung eines langen Dokuments erfordert mehr Arbeit. Zur Einordnung: INT8-KV allein benötigt einschließlich der MTP-Schicht etwa **0.71 GiB bei 64K, 1.42 GiB bei 128K und 2.84 GiB bei 256K**; weitere Gewichte und Arbeitspuffer kommen hinzu. Natives FP8 benötigt in dieser Implementierung ungefähr ebenso viel KV-Speicher wie INT8; sein Vorteil liegt in einem anderen Berechnungspfad und nicht in einem wesentlich kleineren Cache.

Freier VRAM bezeichnet den zum Ladezeitpunkt verfügbaren Speicher, nicht die Zahl auf der GPU-Verpackung. Andere Anwendungen können daher dazu führen, dass ein kleinerer Kontext ausgewählt wird. Falls sich die Kapazität beim erneuten Laden verringert, weist die API eine nicht mehr passende Eingabe zurück, anstatt sie stillschweigend zu kürzen.

Die Quantisierung der GGUF-Gewichte und die KV-Präzision sind voneinander unabhängig. Das ausgewählte UD-IQ3_XXS-Modell hat komprimierte Gewichte; INT8 oder FP8 beschreibt den Unterhaltungscache und seinen Attention-Pfad. KV mit geringerer Präzision kann Antworten verändern, daher versprechen diese Modi keine mit FP32-KV identischen Ausgaben. Die Implementierung behält alle sichtbaren Kontextpositionen und jeden vom Modell ausgewählten Experten bei.

Neue Launcher-Konfigurationen verwenden MTP 3, das mehrere Token im Voraus vorschlägt und sie mit dem Hauptmodell überprüft. Das kann die Generierung beschleunigen, doch der Gewinn hängt davon ab, wie viele Vorschläge akzeptiert werden. Die Überprüfung soll das Dekodierverhalten des Hauptmodells bewahren; Unterschiede in numerischen Ausführungspfaden können dennoch Token verändern. Deshalb verspricht aktiviertes MTP keine mit deaktiviertem MTP identischen Ausgaben.

MTP unterstützt sowohl Greedy-Generierung als auch temperaturbasiertes Sampling, einschließlich Wiederholungs-, Häufigkeits- und Präsenzstrafen. Bei Sampling-Anfragen zieht das Hauptmodell für jede Überprüfungszeile eine Stichprobe, und nur das übereinstimmende Präfix des Entwurfs wird akzeptiert.

Explizite Einstellungen überschreiben die automatische Auswahl, und bereits manuell gewählte Einstellungen bleiben wirksam. Beispielsweise legt `run-api.bat --kv-type int8` INT8 fest; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` stellt die automatische Auswahl von KV, Kontext und Expertenplatzierung wieder her.

## API-Server

**TinyStrata bietet ausschließlich eine API.** Es stellt OpenAI-kompatible APIs für Chat Completions (`/v1/chat/completions`) und Responses (`/v1/responses`) sowie eine Anthropic-kompatible Messages-API (`/v1/messages`) bereit, einschließlich gestreamter Generierung. Verbinden Sie einen API-Client oder ein externes Frontend; TinyStrata enthält keine integrierte Chat-Oberfläche.

Standardmäßig ist Thinking aktiviert, mit `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` und maximal **32,768 Ausgabetoken** einschließlich Thinking. Einstellungen pro Anfrage haben Vorrang.

Die API lauscht standardmäßig unter `http://127.0.0.1:18080`. Um einen anderen Port zu verwenden, führen Sie `run-api.bat --port 18081` im installierten TinyStrata-Ordner aus.

Die API kann starten, bevor die Modelldateien installiert sind. `POST /v1/model/download` mit einem leeren JSON-Body (`{}`) lädt das festgelegte [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS-Sprachmodell](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, einschließlich MTP-Gewichten) und seinen [Vision-Projektor](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB) herunter, insgesamt etwa **15.0 GB**. Die SHA-256-Prüfsummen beider Dateien werden geprüft, der Tokenizer wird extrahiert und die Dateien werden geladen. `GET /v1/model/download` meldet den Fortschritt; unterbrochene Downloads können fortgesetzt werden. Eine wiederholte Download-Anfrage verwendet den aktiven Auftrag weiter, und `POST /v1/model/download/cancel` kann die Übertragung oder Überprüfung abbrechen. Ein abgeschlossener Download bedeutet noch keine Inferenzbereitschaft: Warten Sie, bis der Download-Status `ready` lautet und `/health` sowohl `loaded: true` als auch `images: true` meldet.

Das Installationsprogramm bereitet die Engine und die GPU-Laufzeitbibliotheken vor und lädt beide Modelldateien automatisch herunter. Bestehende Konfigurationen behalten ihr ausgewähltes lokales Modell bei, bis Sie es ausdrücklich ändern.

## Danksagungen

TinyStrata ist ein unabhängiger Fork von [Strata](https://github.com/Niko1221/Strata) und basiert in weiten Teilen auf dessen Implementierung. Vielen Dank an [Niko](https://github.com/Niko1221) und die Strata-Community für die Engine und die Arbeit, die TinyStrata möglich gemacht haben.

Vielen Dank auch an [Aaron Flin](https://github.com/aflin) und das [MoMoE-Projekt](https://github.com/aflin/momoe) für den in TinyStrata angepassten Code für KV-Cache, Attention und quantisierte Experten.

## Lizenz

MIT. Siehe [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Upstream-Attribution: [NOTICE](NOTICE). GGML: [MIT-Lizenz](third_party/ggml/LICENSE).
