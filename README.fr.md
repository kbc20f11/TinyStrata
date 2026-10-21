# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata est un moteur d’inférence rapide en cours de développement pour les modèles MoE de la classe 30B, exclusivement sous Windows et par API.** Il vise principalement les PC équipés de cartes graphiques grand public avec **8–24 GB de VRAM** et disposant de **32 GB de mémoire système au maximum**.

## Pourquoi nous développons TinyStrata

Avec la hausse des prix des composants PC, assembler un PC haut de gamme devient de plus en plus difficile pour beaucoup de personnes.

C’est pourquoi TinyStrata a été créé pour exécuter l’IA localement à grande vitesse sur des PC équipés de cartes graphiques grand public de la classe RTX 3060 ou supérieure. Nous nous concentrons sur l’exécution aussi rapide que possible de modèles MoE relativement petits, de la classe 30B.

Nous visons principalement les PC disposant de **8–24 GB de VRAM et de 32 GB de mémoire système au maximum**. Nous pensons que cette plage correspond aux configurations les plus courantes des PC équipés de cartes graphiques grand public.

Basé sur Strata, TinyStrata cherche à utiliser efficacement la mémoire GPU pour exécuter rapidement des modèles MoE de la classe 30B, même avec une VRAM limitée. Notre objectif est de vous permettre de travailler avec de longues conversations et de longs documents sur votre propre PC.

Le développement se concentre sur l’inférence locale sous Windows et sa mise à disposition par API.

TinyStrata choisit automatiquement les paramètres d’inférence et de mémoire adaptés à votre matériel, pour que vous puissiez commencer à utiliser l’API sans configuration compliquée ni réglages manuels.

## Pourquoi aucune interface intégrée ?

Il existe déjà de nombreuses interfaces et de nombreux environnements d’exécution d’agents compatibles avec les API pour LLM. TinyStrata prenant en charge les formats d’API courants, vous pouvez connecter et utiliser la solution de votre choix. Si la configuration de la connexion vous semble fastidieuse, vous pouvez demander à un agent IA de s’en charger. En confiant l’interface à des logiciels existants, nous pouvons nous concentrer sur le développement du moteur d’inférence et de l’API.

## Modèle privilégié

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reste le modèle par défaut, pour son équilibre entre vitesse, consommation de mémoire et qualité des réponses sur les PC équipés de cartes graphiques grand public. TinyStrata se concentre sur les petits modèles MoE et prend également en charge **Gemma 4 26B-A4B-it et LFM2.5-24B-A2B**.

**Qwen3.6-35B-A3B UD-IQ3_XXS** utilise le [GGUF MTP d’Unsloth fixé](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf). Les deux fichiers totalisent environ **15.0 GB** : modèle de langage, poids MTP compris, de **14.1 GB** et projecteur de vision (`mmproj-F16.gguf`) de **0.9 GB**.

**Gemma 4 26B-A4B-it QAT Q4_0** utilise [le modèle et le projecteur de vision officiels de Google](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) avec [l’assistant MTP QAT correspondant d’Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Les trois fichiers totalisent **15.886 GB** : modèle **14.439 GB**, projecteur **1.195 GB** et assistant **0.252 GB**.

**LFM2.5-24B-A2B Q4_K_M** utilise le [GGUF officiel de LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF). Son unique fichier de modèle de langage pèse **14.53 GB**.

Les trois moteurs utilisent un cache automatique d’experts en VRAM, le calcul d’experts sur CPU depuis la RAM, le streaming des poids à l’entrée et les CUDA Graphs. Ils partagent la même API et permettent de changer de modèle sans redémarrer, avec des paramètres enregistrés séparément pour chaque modèle. Aucun expert sélectionné n’est omis. Ils permettent aussi la croissance automatique de la mémoire KV physique, sans modifier la capacité maximale de contexte.

| Fonction | [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B-it) | [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| --- | --- | --- | --- |
| Contexte maximal | 256K | 256K | 128K |
| Entrée d’images | Oui ; encodeur sur GPU par défaut | Oui ; encodeur sur GPU par défaut | Non |
| MTP | 3 | 3 | Non |
| Formats KV | FP32 / INT8 / FP8 natif | FP16 / INT8 / FP8 natif (expérimental) | FP16 / INT8 / FP8 natif |
| KV automatique (par défaut) | FP8 → INT8 | INT8 | FP8 → INT8 |

Les en-têtes renvoient aux fiches officielles des modèles. **FP8 → INT8** désigne un KV FP8 et une Attention native sur Tensor Cores FP8 si le GPU et les kernels compilés le permettent, sinon INT8. La précision KV est indépendante de la quantification des poids GGUF ; une précision KV plus faible peut modifier la sortie par rapport à FP32 (Qwen) ou FP16 (Gemma/LFM).

Gemma utilise INT8 par défaut. Le FP8 natif est expérimental et peut être sélectionné explicitement avec `run-api.bat --model-family gemma4 --kv-type fp8-mma`.

La **capacité de contexte automatique** dépend de la VRAM totale du GPU, comme indiqué ci-dessous ; la VRAM libre détermine la taille du cache d’experts. Une valeur explicite de `--max-context N` est prioritaire.

| VRAM totale (exemples) | Qwen3.6 / Gemma 4 | LFM2.5 |
| --- | ---: | ---: |
| 8 GB | 64K | 64K |
| 12 GB | 128K | 128K |
| 16 GB ou plus | 256K | 128K |

Les classes de capacité tiennent compte des différences signalées par le pilote : une carte de 16 GB indiquant 15.x GiB sélectionne donc toujours 256K pour Qwen et Gemma.

Si l’exécutable d’un modèle supplémentaire manque, consultez les [instructions de compilation manuelle](README.md#model-focus).

## Installation

Exécutez cette commande dans PowerShell 64 bits pour installer TinyStrata et démarrer l’API à l’adresse `http://127.0.0.1:18080/v1` :

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Pour relancer TinyStrata après un redémarrage de Windows, exécutez cette commande ou double-cliquez sur le même fichier `run-api.bat` :

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

Vous pouvez aussi donner à un agent IA le lien vers le [dépôt GitHub de TinyStrata](https://github.com/kbc20f11/TinyStrata) et lui confier l’installation.

### Changer de modèle sans redémarrer

Le programme d’installation démarre Qwen par défaut. Pour préparer Gemma sur le même serveur API, lancez son téléchargement et consultez la progression :

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

Répétez la consultation jusqu’à `state: ready`, puis demandez explicitement le changement. Télécharger un modèle inactif le prépare sans l’activer :

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

Pour démarrer directement avec Gemma, exécutez `run-api.bat --model-family gemma4` ; l’API utilise `http://127.0.0.1:18082/v1`. Pour LFM, remplacez `gemma-4-26b-a4b-it` par `lfm2.5-24b-a2b` dans les requêtes de téléchargement, de progression et de changement ci-dessus, ou exécutez `run-api.bat --model-family lfm2` pour le démarrer directement sur `http://127.0.0.1:18081/v1`.

Consultez `GET /v1/model/switch` jusqu’à `state: ready`. Pour revenir à Qwen, envoyez la même requête avec `{"model":"qwen3.6-35b-a3b"}`. Le démarrage d’un téléchargement ou d’un changement renvoie `202` ; un changement pendant une génération renvoie `409`, réessayez après sa fin. Les requêtes d’inférence pendant le changement reçoivent `503`. Le port et l’authentification restent inchangés. La sélection vaut pour la session actuelle du serveur ; au redémarrage, le modèle choisi dans le lanceur est utilisé.

## Vitesse mesurée — Qwen3.6

Mesuré le **2026-10-21** sous Windows avec une RTX PRO 4000 Blackwell SFF Edition (24 GB), un Ryzen 9 7945HX3D, 64 GB de RAM et le modèle Unsloth MTP UD-IQ3_XXS fixé pour ce projet. L’encodeur d’images est resté sur GPU pendant les requêtes texte par API. Le raisonnement, les valeurs d’échantillonnage par défaut du modèle (température 0.7) et MTP 3 étaient activés. Les blocs d’entrée et la répartition des experts étaient automatiques ; la capacité de contexte était fixée selon la classe de VRAM. Le chemin hybride CUDA Graph est resté désactivé par défaut et les CUDA Graphs habituels en exécution entièrement sur GPU sont restés actifs.

| Mémoire disponible | Capacité de contexte | Génération INT8 | Génération FP8 native | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 117.80 tok/s | 131.34 tok/s | Tous les poids résidents en VRAM |
| environ 16 GB | 256K | 98.90 tok/s | 101.68 tok/s | 8 GiB occupés séparément ; experts calculés sur CPU et GPU |
| environ 12 GB | 128K | 87.99 tok/s | 86.66 tok/s | 12 GiB occupés séparément ; experts calculés sur CPU et GPU |
| environ 8 GB | 64K | 60.69 tok/s | 60.69 tok/s | FP8 : pas de cache d’experts sur le GPU |

Les mêmes trois textes ont été exécutés deux fois par configuration FP8/INT8 avec les valeurs d’échantillonnage par défaut du modèle et 128 tokens générés par requête, raisonnement compris. La vitesse est le nombre total de tokens divisé par le temps total de décodage, hors chargement et traitement de l’entrée ; le contexte indique la capacité du cache. Les lignes avec moins de mémoire réservent 8, 12 ou 16 GiB sur le même GPU de 24 GB et utilisent le contexte de cette classe de VRAM ; ce ne sont pas des mesures de GPU physiques de 16/12/8 GB.

## Vitesse mesurée — Gemma 4

Mesuré sur le même PC Windows avec 64 GB de RAM, Gemma 4 26B-A4B-it QAT Q4_0 et son assistant MTP QAT correspondant. L’encodeur d’images est resté sur GPU pendant les requêtes texte par API. MTP 3, les CUDA Graphs et les valeurs de raisonnement et d’échantillonnage propres au modèle étaient activés ; la capacité de contexte était fixée selon la classe de VRAM, tandis que les blocs d’entrée et la répartition des experts étaient automatiques.

Le FP8 natif et INT8 ont été mesurés le **2026-10-21**.

| VRAM disponible | Capacité de contexte | INT8 (par défaut) | FP8 natif (expérimental) | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 113.20 tok/s | 112.31 tok/s | INT8 / FP8: Calcul des experts sur GPU |
| Environ 16 GB | 256K | 100.08 tok/s | 96.67 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |
| Environ 12 GB | 128K | 81.66 tok/s | 82.08 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |
| Environ 8 GB | 64K | 51.54 tok/s | 53.41 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |

Les mêmes trois textes ont été exécutés deux fois par configuration FP8/INT8 avec les valeurs d’échantillonnage par défaut du modèle et 128 tokens générés par requête, raisonnement compris. La vitesse est le nombre total de tokens divisé par le temps total de décodage, hors chargement et traitement de l’entrée ; le contexte indique la capacité du cache. Les lignes avec moins de mémoire réservent 8, 12 ou 16 GiB sur le même GPU de 24 GB et utilisent le contexte de cette classe de VRAM ; ce ne sont pas des mesures de GPU physiques de 16/12/8 GB.

## Vitesse mesurée — LFM2.5

Mesuré sur le même PC Windows avec 64 GB de RAM et le LFM2.5-24B-A2B Q4_K_M officiel, sans MTP ni Vision. Les CUDA Graphs et les valeurs de raisonnement et d’échantillonnage propres au modèle étaient activés ; la capacité de contexte était fixée selon la classe de VRAM, tandis que les blocs d’entrée et la répartition des experts étaient automatiques.

Le FP8 natif et INT8 ont été mesurés le **2026-10-21**.

| VRAM disponible | Capacité de contexte | INT8 | FP8 natif (par défaut) | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 128K | 140.58 tok/s | 141.80 tok/s | INT8 / FP8: Calcul des experts sur GPU |
| Environ 16 GB | 128K | 135.63 tok/s | 138.21 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |
| Environ 12 GB | 128K | 110.11 tok/s | 114.72 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |
| Environ 8 GB | 64K | 71.10 tok/s | 73.92 tok/s | INT8 / FP8: Calcul des experts sur GPU et CPU |

Les mêmes trois textes ont été exécutés deux fois par configuration FP8/INT8 avec les valeurs d’échantillonnage par défaut du modèle et 128 tokens générés par requête, raisonnement compris. La vitesse est le nombre total de tokens divisé par le temps total de décodage, hors chargement et traitement de l’entrée ; le contexte indique la capacité du cache. Les lignes avec moins de mémoire réservent 8, 12 ou 16 GiB sur le même GPU de 24 GB et utilisent le contexte de cette classe de VRAM ; ce ne sont pas des mesures de GPU physiques de 16/12/8 GB.

## Quel projet choisir ?

| Votre objectif et votre matériel | Projet |
| --- | --- |
| Vous souhaitez une inférence Qwen3.6 / Gemma 4 / LFM2.5 exclusivement par API sous **Windows**, avec **8–24 GB de VRAM et 32 GB de mémoire système au maximum**, par exemple sur un PC équipé d’une carte graphique grand public. | TinyStrata (**Windows uniquement**, en développement) |
| Vous disposez à la fois **d’un GPU de la classe RTX 3060 ou supérieure** et **de 64 GB de mémoire système ou plus**. | Nous recommandons [Strata (projet amont)](https://github.com/Niko1221/Strata). |
| Vous souhaitez une **interface de chat intégrée**. | Nous recommandons [Strata (projet amont)](https://github.com/Niko1221/Strata). |

## Répartition de la mémoire — Qwen3.6

Un modèle MoE contient de nombreux **experts** : des réseaux neuronaux plus petits sélectionnés pour chaque token. TinyStrata conserve autant d’experts que possible dans la mémoire GPU (VRAM). Pendant la génération, il peut calculer les experts sélectionnés qui ne sont pas en cache sur le CPU, en utilisant la mémoire système. Lors du traitement de lots d’entrée plus volumineux, il peut à la place transférer les poids des experts depuis la RAM via des tampons GPU temporaires. Les experts sélectionnés par le modèle restent inclus dans le calcul.

Les fichiers du modèle téléchargés représentent environ **15.0 GB** au total : **14.1 GB** pour le modèle de langage (poids MTP compris) et **0.9 GB** pour le projecteur de vision (`mmproj-F16.gguf`). Le modèle de langage n’a pas besoin de tenir entièrement dans la VRAM. La mémoire GPU doit également accueillir l’encodeur de vision et les tampons de traitement d’image, le **cache KV** de la conversation, les autres poids du modèle et les tampons de calcul temporaires :

![Répartition de la mémoire entre la VRAM du GPU et la mémoire système dans TinyStrata](assets/memory-sharing.png)

Il s’agit d’un schéma de répartition avec le traitement d’image par défaut sur GPU, pas d’un partitionnement de taille fixe. À chaque chargement du modèle, TinyStrata démarre et préchauffe l’encodeur de vision sur le GPU avant de charger le modèle de langage ; ses poids et ses tampons de travail préchauffés restent résidents. Il alloue ensuite les premières pages KV, les autres poids et les tampons de travail, puis dimensionne le cache d’experts selon la VRAM libre restante. Sous Windows, il utilise la plus petite valeur entre la mémoire libre indiquée par CUDA et la VRAM dédiée libre indiquée par le pilote NVIDIA, lorsque les deux sont disponibles.

Le contenu du cache change selon les experts utilisés. Avec la croissance KV par défaut, les conversations courtes laissent davantage de VRAM aux experts ; lorsque la séquence grandit, le cache d’experts rend de la mémoire à KV. Les nouvelles requêtes plus courtes peuvent rendre les pages KV inutilisées au cache d’experts. `run-api.bat --kv-whole` alloue toute la capacité KV dès le départ ; `run-api.bat --kv-grow` rétablit la croissance. Les autres applications peuvent toujours consommer de la VRAM.

Le placement des experts vise normalement une marge de 768 MiB. Dans le mode `--kv-whole` de Qwen, comme dans Strata, lorsque tous les experts du modèle principal sont près de tenir en VRAM, il peut libérer les copies compactées des poids Q8, réduire un tampon de scores Attention s’il existe et ramener la marge à 256 MiB ; les experts MTP peuvent fonctionner sur le CPU. `--vram-reserve-mib N` fixe explicitement la réserve. Il s’agit d’un objectif au moment du placement, pas d’une garantie permanente de mémoire libre.

Davantage d’experts résidents sur le GPU signifie généralement moins de travail pour le CPU. En mode hybride, la mémoire système conserve également des copies des experts mis en cache dans la VRAM ; ajouter de la VRAM ne réduit donc pas l’utilisation de la RAM octet pour octet. Laissez aussi de la RAM au système d’exploitation et aux autres applications. Une plus grande **capacité de RAM** permet de conserver les données du modèle, mais la vitesse dépend également du CPU, de la bande passante de la RAM et du GPU. La « mémoire GPU partagée » de Windows diffère du calcul délibéré des experts sur le CPU par TinyStrata.

Le cache KV reste dans la VRAM ; la mémoire système contient les données des experts pour ce mécanisme de déport. Une longue conversation a donc toujours besoin de suffisamment de mémoire GPU pour son cache KV.

## Paramètres automatiques — Qwen3.6

Les nouvelles configurations choisissent automatiquement KV/Attention et la capacité de contexte. Vous n’avez pas besoin de sélectionner un profil pour 8 GB, 12 GB ou 16 GB.

Pour Qwen3.6, une RTX 3060 sélectionne INT8. Les environnements RTX 40/50 pris en charge peuvent sélectionner le FP8 natif ; le moteur vérifie les noyaux compilés et les capacités du GPU ainsi que la génération matérielle. Si le FP8 natif n’est pas disponible, le mode automatique utilise INT8. Cela sélectionne un chemin pris en charge sans exécuter de benchmark de vitesse ou de qualité au démarrage.

Le moteur natif estime la taille des blocs d’entrée au démarrage, en choisissant 512, 1024, 2048 ou 4096 tokens selon la VRAM libre, la capacité de contexte et la possibilité de placer les experts sur le GPU. Des blocs plus grands peuvent réduire les transferts répétés de poids de la RAM vers le GPU pendant le traitement des entrées. `run-api.bat --chunk N` fixe la taille ; `run-api.bat --chunk auto` rétablit la sélection automatique native.

La capacité de contexte est la limite logique de l’entrée et de la sortie générée réunies. Par défaut, la mémoire KV physique est allouée à mesure que la conversation grandit, au lieu de réserver toute la capacité pour une conversation courte. Les documents longs nécessitent toujours davantage de calcul et, à terme, davantage de mémoire KV.

À pleine capacité, le cache KV INT8 de Qwen occupe environ **0.71 GiB à 64K, 1.42 GiB à 128K et 2.84 GiB à 256K**, couche MTP comprise ; les autres poids et tampons de travail s’y ajoutent. Dans cette implémentation, le FP8 natif utilise approximativement le même espace de stockage KV que l’INT8 ; son intérêt réside dans un chemin de calcul différent, plutôt que dans un cache nettement plus petit.

La capacité de contexte est choisie selon la VRAM totale du GPU ; le cache d’experts est ensuite ajusté à la VRAM libre restante au chargement. Les autres applications qui utilisent de la VRAM laissent moins de place aux experts en cache sur GPU. L’API rejette les entrées qui dépassent la capacité de contexte au lieu de les tronquer silencieusement.

La quantification des poids GGUF et la précision KV sont distinctes. Le modèle UD-IQ3_XXS sélectionné possède des poids compressés ; INT8 ou FP8 décrit le cache de conversation et son chemin Attention. Une précision KV plus faible peut modifier les réponses ; ces modes ne promettent donc pas une sortie identique à celle du cache KV FP32. L’implémentation conserve toutes les positions de contexte visibles et chaque expert sélectionné par le modèle.

Les nouvelles configurations du lanceur utilisent MTP 3, qui propose plusieurs tokens à l’avance et les vérifie avec le modèle principal. Cela peut accélérer la génération, mais le gain dépend du nombre de propositions acceptées. La vérification vise à préserver le comportement de décodage du modèle principal ; des différences dans les chemins d’exécution numérique peuvent néanmoins modifier les tokens. Activer MTP ne garantit donc pas une sortie identique à celle obtenue en le désactivant.

MTP prend en charge la génération gloutonne et l’échantillonnage fondé sur la température, y compris les pénalités de répétition, de fréquence et de présence. Pour les requêtes avec échantillonnage, le modèle principal échantillonne chaque ligne de vérification, et seul le préfixe concordant de la proposition est accepté.

Les paramètres explicites remplacent les choix automatiques, et les paramètres déjà sélectionnés manuellement restent en vigueur. Par exemple, `run-api.bat --kv-type int8` impose INT8 ; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` rétablit la sélection automatique de KV, du contexte et du placement des experts.

## Serveur API

**TinyStrata fonctionne exclusivement par API.** Il fournit des API Chat Completions (`/v1/chat/completions`) et Responses (`/v1/responses`) compatibles avec OpenAI ainsi qu’une API Messages (`/v1/messages`) compatible avec Anthropic, avec génération en streaming. Connectez un client API ou une interface externe ; TinyStrata ne comprend pas d’interface de chat intégrée.

Les nouvelles configurations utilisent des paramètres d’échantillonnage propres à chaque modèle. Les valeurs de chaque requête sont prioritaires ; les réglages personnalisés enregistrés restent inchangés.

| Paramètre | Qwen3.6 | Gemma 4 | LFM2.5 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.7 | 1.0 | 0.1 |
| `top_p` | 0.95 | 0.95 | 1.0 |
| `top_k` | 20 | 64 | 50 |
| `min_p` | 0.0 | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.05 |
| `enable_thinking` | true | true | true (toujours activé) |
| `max_tokens` (raisonnement compris) | 32768 | 32768 | 32768 |

L’API écoute par défaut sur `http://127.0.0.1:18080`. Pour utiliser un autre port, exécutez `run-api.bat --port 18081` depuis le dossier d’installation de TinyStrata.

Pour Qwen, `run-api.bat --vision gpu` traite les images sur GPU (valeur par défaut des nouvelles configurations) ; `--vision cpu` utilise le CPU et `--vision no` désactive les images. `--vision-tokens N` fixe le nombre maximal de tokens d’image (4096 par défaut). Ces choix sont enregistrés et conservés lors des prochains démarrages si ces options sont omises.

L’API peut démarrer avant l’installation des fichiers du modèle. Lorsque Qwen est sélectionné, `POST /v1/model/download` avec un corps JSON vide (`{}`) télécharge le [modèle de langage Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) fixé pour ce projet (14.1 GB, poids MTP compris) et son [projecteur de vision](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), soit environ **15.0 GB au total**. Il vérifie les sommes de contrôle SHA-256 des deux fichiers, extrait le tokenizer et les charge. `GET /v1/model/download` indique la progression ; les téléchargements interrompus peuvent reprendre. Une nouvelle requête de téléchargement réutilise la tâche active, et `POST /v1/model/download/cancel` peut annuler le transfert ou la vérification. La fin du téléchargement ne signifie pas que l’inférence est prête : attendez que l’état du téléchargement passe à `ready` et que `/health` indique `loaded: true` et, si les images sont activées, `images: true`.

Le programme d’installation prépare les exécutables des moteurs et les bibliothèques d’exécution GPU, puis télécharge Qwen et son projecteur de vision par défaut. Les configurations existantes conservent le modèle local sélectionné jusqu’à ce que vous le changiez explicitement.

## Remerciements

TinyStrata est un fork indépendant de [Strata](https://github.com/Niko1221/Strata) et s’appuie largement sur son implémentation. Merci à [Niko](https://github.com/Niko1221) et à la communauté Strata pour le moteur et le travail qui ont rendu TinyStrata possible.

Merci également à [Aaron Flin](https://github.com/aflin) et au [projet MoMoE](https://github.com/aflin/momoe) pour le code du cache KV, d’attention et des experts quantifiés adapté dans TinyStrata.

## Licence

MIT. Voir [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Attribution au projet amont : [NOTICE](NOTICE). GGML : [licence MIT](third_party/ggml/LICENSE).
