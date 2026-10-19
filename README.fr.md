# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata est un moteur d’inférence rapide en cours de développement pour les modèles MoE de la classe 30B, exclusivement sous Windows et par API.** Il vise principalement les PC équipés de cartes graphiques grand public avec **8–24 GB de VRAM** et disposant de **32 GB de mémoire système au maximum**.

## Pourquoi nous développons TinyStrata

Avec la hausse des prix des composants PC, assembler un PC haut de gamme devient de plus en plus difficile pour beaucoup de personnes.

C’est pourquoi TinyStrata a été créé pour exécuter l’IA localement à grande vitesse sur des PC équipés de cartes graphiques grand public de la classe RTX 3060 ou supérieure. Nous nous concentrons sur l’exécution aussi rapide que possible de modèles MoE relativement petits, de la classe 30B.

Nous visons principalement les PC disposant de **8–24 GB de VRAM et de 32 GB de mémoire système au maximum**. Nous pensons que cette plage correspond aux configurations les plus courantes des PC équipés de cartes graphiques grand public.

Basé sur Strata, TinyStrata vise à accélérer l’inférence de Qwen3.6-35B-A3B dans les environnements où la VRAM est limitée. Nous souhaitons utiliser efficacement la mémoire GPU pour vous permettre de travailler avec de longues conversations et de longs documents sur votre propre PC.

TinyStrata concentre son développement sur l’inférence locale rapide sous Windows et sa mise à disposition par API. Cette orientation guide notre implémentation et nos tests.

Nous souhaitons également rendre la prise en main aussi simple que possible. TinyStrata choisit automatiquement les paramètres d’inférence et de mémoire adaptés à votre matériel, pour que vous puissiez utiliser l’API sans configuration compliquée ni réglages manuels.

## Pourquoi aucune interface intégrée ?

Il existe déjà de nombreuses interfaces et de nombreux environnements d’exécution d’agents compatibles avec les API pour LLM. TinyStrata prenant en charge les formats d’API courants, vous pouvez connecter et utiliser la solution de votre choix. Si la configuration de la connexion vous semble fastidieuse, vous pouvez demander à un agent IA de s’en charger. En confiant l’interface à des logiciels existants, nous pouvons nous concentrer sur le développement du moteur d’inférence et de l’API.

## Modèle privilégié

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reste le modèle par défaut, pour son équilibre entre vitesse, consommation de mémoire et qualité des réponses sur les PC équipés de cartes graphiques grand public. TinyStrata se concentre sur les petits modèles MoE et prend également en charge **Gemma 4 26B-A4B-it et LFM2.5-24B-A2B**.

**Gemma 4 26B-A4B-it QAT Q4_0** utilise [le modèle cible et le projecteur de vision officiels de Google](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) avec [l’assistant MTP QAT Q4_0 correspondant d’Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Les trois fichiers totalisent **15.886 GB** : cible 14.439 GB, projecteur 1.195 GB et assistant 0.252 GB. Le moteur prend en charge l’encodeur d’images résident sur GPU, un cache adaptatif d’experts en VRAM, leur calcul sur CPU depuis la RAM et MTP 3 via la même API. Aucun expert sélectionné n’est omis. Gemma accepte un **KV FP16 ou INT8** ; le choix automatique actuel est **FP16**. L’entrée d’images, MTP 3, l’utilisation conjointe de la RAM et le changement de modèle ont été testés avec le modèle réel.

LFM utilise le [GGUF Q4_K_M officiel de LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (**14.53 GB**), un **KV FP16** et jusqu’à **128K de contexte**, sans MTP ni Vision. Il bénéficie du cache automatique d’experts en VRAM, du calcul d’experts sur CPU, du streaming des poids à l’entrée et des CUDA Graphs ; aucun expert sélectionné n’est omis. Ses valeurs par défaut sont `temperature=0.1`, `top_k=50` et `repetition_penalty=1.05`, avec le raisonnement toujours activé. Les paramètres Qwen, Gemma et LFM sont conservés séparément. Le moteur LFM est encore en cours de validation et ne garantit pas les mêmes tokens que llama.cpp ou entre différentes répartitions CPU/GPU.

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

Les mesures suivantes ont été effectuées le 2026-10-19 sous Windows avec une RTX PRO 4000 Blackwell SFF Edition (24 GB), un Ryzen 9 7945HX3D et le modèle Unsloth MTP UD-IQ3_XXS fixé pour ce projet. Le modèle et le projecteur de vision sur GPU étaient chargés et préchauffés. Les paramètres étaient ceux de l’API par défaut : réflexion activée, échantillonnage à température 0.7, MTP 3, blocs de prompt de 512 tokens, placement des experts et sélection du contexte automatiques. Le chemin hybride CUDA Graph optionnel était désactivé par défaut ; les CUDA Graphs habituels en exécution entièrement sur GPU restaient actifs. Sur ce GPU, `auto` sélectionnait le FP8 natif ; seule la précision KV était imposée pour la colonne INT8.

**Des mesures sur des machines équipées de GPU physiques de 8 GB, 12 GB et 16 GB sont prévues ultérieurement.** Les lignes à mémoire réduite du tableau ci-dessous ont été mesurées en limitant la mémoire disponible sur le même GPU de 24 GB et ne sont pas des benchmarks de GPU physiques de 8/12/16 GB.

| Mémoire disponible | Capacité de contexte | Génération INT8 | Génération FP8 native | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | Tous les poids résidents en VRAM |
| environ 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | 8 GiB occupés séparément ; experts calculés sur CPU et GPU |
| environ 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | 12 GiB occupés séparément ; experts calculés sur CPU et GPU |
| environ 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | 16 GiB occupés séparément ; experts calculés sur CPU et GPU |

## Vitesse mesurée — Gemma 4

Mesuré le 2026-10-19 sur le même PC Windows avec RTX PRO 4000 Blackwell SFF Edition (24 GB), Ryzen 9 7945HX3D et 64 GB de RAM. Les modèles utilisés sont Gemma 4 26B-A4B-it QAT Q4_0 de Google et l’assistant MTP QAT correspondant d’Unsloth, avec l’encodeur d’images résident sur GPU, des entrées texte par API, MTP 3 et les CUDA Graphs activés. Les valeurs par défaut de Google sont conservées : réflexion activée, `temperature=1.0`, `top_p=0.95`, `top_k=64`. Le contexte et la répartition des experts sont choisis automatiquement, avec des blocs d’entrée automatiques de 1120 tokens.

| VRAM disponible | Capacité de contexte | Génération FP16 | Génération INT8 | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 105.56 tok/s | — | Calcul des experts sur GPU et CPU ; INT8 non mesuré |
| Environ 8 GB | 64K | 48.22 tok/s | 48.97 tok/s | Calcul des experts sur CPU pendant la génération |

Trois entrées fixes ont été exécutées une fois chacune, avec 128 tokens de sortie, réflexion comprise. La vitesse correspond au total des tokens divisé par le temps total de décodage, hors chargement et traitement des entrées. La condition de 8 GB est obtenue en réservant séparément 16 GiB sur la même carte de 24 GB ; il ne s’agit pas d’un test d’une carte physique de 8 GB. 64K/256K indique la capacité du cache, pas la longueur des entrées.

## Vitesse mesurée — LFM2.5 (expérimental)

Mesures sur le même PC Windows avec 64 GB de mémoire système, le LFM2.5-24B-A2B Q4_K_M officiel, un KV FP16, sans MTP ni Vision, avec CUDA Graphs activé. Les valeurs actuelles de l’API ont été utilisées : raisonnement activé, `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`. La capacité de contexte, les blocs d’entrée (256 tokens dans tous les cas) et la répartition des experts ont été choisis automatiquement.

| VRAM disponible | Capacité de contexte | Génération FP16 | Remarques |
| --- | ---: | ---: | --- |
| 24 GB | 128K | 151.71 tok/s | Tous les poids résidents en VRAM |
| Environ 16 GB | 128K | 111.71 tok/s | Calcul des experts sur GPU et CPU |
| Environ 12 GB | 128K | 70.80 tok/s | Calcul des experts sur GPU et CPU |
| Environ 8 GB | 64K | 51.20 tok/s | Calcul des experts sur GPU et CPU |

Les trois mêmes textes d’entrée que pour Qwen ont été exécutés deux fois chacun, avec 128 tokens générés par requête, raisonnement compris. La vitesse est le nombre total de tokens générés divisé par le temps total de décodage, hors chargement et traitement de l’entrée. Les lignes à mémoire réduite limitent la VRAM disponible sur la même carte de 24 GB ; ce ne sont pas des mesures sur des GPU physiques de 8/12/16 GB. Le modèle et les paramètres d’échantillonnage diffèrent du tableau Qwen.

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

Il s’agit d’un schéma de répartition avec le traitement d’image par défaut sur GPU, pas d’un partitionnement de taille fixe. À chaque chargement du modèle, TinyStrata démarre et préchauffe l’encodeur de vision sur le GPU avant de charger le modèle de langage ; ses poids et ses tampons de travail préchauffés restent résidents. Il alloue ensuite le cache KV, les autres poids et les tampons de travail, puis dimensionne le cache d’experts selon la VRAM libre restante. Sous Windows, il utilise la plus petite valeur entre la mémoire libre indiquée par CUDA et la VRAM dédiée libre indiquée par le pilote NVIDIA, lorsque les deux sont disponibles.

Le contenu du cache change selon les experts utilisés ; sa capacité est recalculée au rechargement du modèle. Les autres applications ouvertes par la suite peuvent toujours consommer de la VRAM.

Le placement des experts vise normalement une marge de 768 MiB. Comme Strata, lorsque tous les experts du modèle principal sont près de tenir en VRAM, il peut libérer les copies compactées des poids Q8, réduire un tampon de scores Attention s’il existe et ramener la marge à 256 MiB ; les experts MTP peuvent fonctionner sur le CPU. `--vram-reserve-mib N` fixe explicitement la réserve. Il s’agit d’un objectif au moment du placement, pas d’une garantie permanente de mémoire libre.

Davantage d’experts résidents sur le GPU signifie généralement moins de travail pour le CPU. En mode hybride, la mémoire système conserve également des copies des experts mis en cache dans la VRAM ; ajouter de la VRAM ne réduit donc pas l’utilisation de la RAM octet pour octet. Laissez aussi de la RAM au système d’exploitation et aux autres applications. Une plus grande **capacité de RAM** permet de conserver les données du modèle, mais la vitesse dépend également du CPU, de la bande passante de la RAM et du GPU. La « mémoire GPU partagée » de Windows diffère du calcul délibéré des experts sur le CPU par TinyStrata.

Le cache KV reste dans la VRAM ; la mémoire système contient les données des experts pour ce mécanisme de déport. Une longue conversation a donc toujours besoin de suffisamment de mémoire GPU pour son cache KV.

## Paramètres automatiques — Qwen3.6

Les nouvelles configurations choisissent automatiquement KV/Attention et la capacité de contexte. Vous n’avez pas besoin de sélectionner un profil pour 8 GB, 12 GB ou 16 GB.

Une RTX 3060 sélectionne INT8. Les environnements RTX 40/50 pris en charge peuvent sélectionner le FP8 natif ; le moteur vérifie les noyaux compilés et les capacités du GPU ainsi que la génération matérielle. Si le FP8 natif n’est pas disponible, le mode automatique utilise INT8. Cela sélectionne un chemin pris en charge sans exécuter de benchmark de vitesse ou de qualité au démarrage. **L’Attention FP8 native reste expérimentale.**

Le moteur natif estime la taille des blocs d’entrée au démarrage, en choisissant 512, 1024, 2048 ou 4096 tokens selon la VRAM libre, la capacité de contexte et la possibilité de placer les experts sur le GPU. Des blocs plus grands peuvent réduire les transferts répétés de poids de la RAM vers le GPU pendant le traitement des entrées. `run-api.bat --chunk N` fixe la taille ; `run-api.bat --chunk auto` rétablit la sélection automatique native.

La capacité de contexte est l’espace maximal réservé à l’entrée et à la sortie générée réunies. Une capacité plus élevée utilise davantage de mémoire KV même avec une conversation courte, ce qui laisse moins de VRAM pour les experts. La capacité n’implique pas non plus une vitesse identique pour toutes les longueurs de conversation : traiter effectivement un long document demande davantage de travail. À titre indicatif, le cache KV INT8 seul occupe environ **0.71 GiB à 64K, 1.42 GiB à 128K et 2.84 GiB à 256K**, couche MTP comprise ; les autres poids et tampons de travail s’y ajoutent. Dans cette implémentation, le FP8 natif utilise approximativement le même espace de stockage KV que l’INT8 ; son intérêt réside dans un chemin de calcul différent, plutôt que dans un cache nettement plus petit.

La VRAM libre désigne la mémoire disponible au moment du chargement, pas le chiffre imprimé sur la boîte du GPU. Les autres applications peuvent donc entraîner la sélection d’un contexte plus petit. Si un rechargement réduit la capacité, l’API rejette une entrée qui ne tient plus au lieu de la tronquer silencieusement. Sous Windows, la sélection automatique du contexte consulte la mémoire libre de CUDA et la valeur globale du périphérique indiquée par le pilote NVIDIA, puis retient la plus petite si les deux requêtes aboutissent.

La quantification des poids GGUF et la précision KV sont distinctes. Le modèle UD-IQ3_XXS sélectionné possède des poids compressés ; INT8 ou FP8 décrit le cache de conversation et son chemin Attention. Une précision KV plus faible peut modifier les réponses ; ces modes ne promettent donc pas une sortie identique à celle du cache KV FP32. L’implémentation conserve toutes les positions de contexte visibles et chaque expert sélectionné par le modèle.

Les nouvelles configurations du lanceur utilisent MTP 3, qui propose plusieurs tokens à l’avance et les vérifie avec le modèle principal. Cela peut accélérer la génération, mais le gain dépend du nombre de propositions acceptées. La vérification vise à préserver le comportement de décodage du modèle principal ; des différences dans les chemins d’exécution numérique peuvent néanmoins modifier les tokens. Activer MTP ne garantit donc pas une sortie identique à celle obtenue en le désactivant.

MTP prend en charge la génération gloutonne et l’échantillonnage fondé sur la température, y compris les pénalités de répétition, de fréquence et de présence. Pour les requêtes avec échantillonnage, le modèle principal échantillonne chaque ligne de vérification, et seul le préfixe concordant de la proposition est accepté.

Les paramètres explicites remplacent les choix automatiques, et les paramètres déjà sélectionnés manuellement restent en vigueur. Par exemple, `run-api.bat --kv-type int8` impose INT8 ; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` rétablit la sélection automatique de KV, du contexte et du placement des experts.

## Serveur API

**TinyStrata fonctionne exclusivement par API.** Il fournit des API Chat Completions (`/v1/chat/completions`) et Responses (`/v1/responses`) compatibles avec OpenAI ainsi qu’une API Messages (`/v1/messages`) compatible avec Anthropic, avec génération en streaming. Connectez un client API ou une interface externe ; TinyStrata ne comprend pas d’interface de chat intégrée.

Pour Qwen, le raisonnement est activé par défaut, avec `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` et un maximum de **32,768 tokens de sortie**, raisonnement compris. Les paramètres définis pour chaque requête sont prioritaires.

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
