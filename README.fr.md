# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata est un moteur d’inférence rapide en cours de développement pour les modèles MoE de la classe 30B, exclusivement sous Windows et par API.** Il vise principalement les PC de jeu courants et les ordinateurs portables de jeu dotés d’un GPU avec **8–24 GB de VRAM**, ainsi que de **32 GB de mémoire système au maximum**.

## Pourquoi nous développons TinyStrata

Avec la hausse des prix des composants PC, assembler un PC haut de gamme devient de plus en plus difficile pour beaucoup de personnes.

C’est pourquoi TinyStrata a été créé pour exécuter l’IA localement à grande vitesse sur des PC de jeu courants et des ordinateurs portables de jeu équipés d’un GPU de la classe RTX 3060 ou supérieure. Nous nous concentrons sur l’exécution aussi rapide que possible de modèles MoE relativement petits, de la classe 30B.

Nous visons principalement les PC disposant de **8–24 GB de VRAM et de 32 GB de mémoire système au maximum**. Nous pensons que cette plage correspond aux configurations les plus courantes des PC de jeu et des ordinateurs portables de jeu.

Basé sur Strata, TinyStrata vise à accélérer l’inférence de Qwen3.6-35B-A3B dans les environnements où la VRAM est limitée. Nous souhaitons utiliser efficacement la mémoire GPU pour vous permettre de travailler avec de longues conversations et de longs documents sur votre propre PC.

TinyStrata concentre son développement sur l’inférence locale rapide sous Windows et sa mise à disposition par API. Cette orientation guide notre implémentation et nos tests.

Nous souhaitons également rendre la prise en main aussi simple que possible. TinyStrata choisit automatiquement les paramètres d’inférence et de mémoire adaptés à votre matériel, pour que vous puissiez utiliser l’API sans configuration compliquée ni réglages manuels.

## Pourquoi aucune interface intégrée ?

Il existe déjà de nombreuses interfaces et de nombreux environnements d’exécution d’agents compatibles avec les API pour LLM. TinyStrata prenant en charge les formats d’API courants, vous pouvez connecter et utiliser la solution de votre choix. Si la configuration de la connexion vous semble fastidieuse, vous pouvez demander à un agent IA de s’en charger. En confiant l’interface à des logiciels existants, nous pouvons nous concentrer sur le développement du moteur d’inférence et de l’API.

## Modèle privilégié

TinyStrata se concentre actuellement exclusivement sur [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Nous envisagerons de prendre en charge un autre modèle lorsqu’un modèle MoE sera mieux adapté que Qwen3.6-35B-A3B à l’inférence locale rapide sur des PC de jeu courants et des ordinateurs portables de jeu.

À nos yeux, le problème est que le développement et la publication de grands modèles progressent, tandis que le choix de modèles MoE de la classe 30B faciles à utiliser sur des PC de jeu courants ne s’élargit pas suffisamment. [Qwen3.6-35B-A3B, publié le 16 avril 2026](https://github.com/QwenLM/Qwen3.8#news), reste adapté à notre usage malgré le temps écoulé depuis sa publication : nous privilégions l’équilibre entre vitesse, faible consommation de mémoire et qualité des réponses. Nous n’avons encore trouvé aucun petit modèle MoE qui, selon notre évaluation, offre un équilibre nettement supérieur dans les environnements que nous visons.

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

## Vitesse mesurée

Les mesures suivantes ont été effectuées le 2026-10-17 sous Windows avec une RTX PRO 4000 Blackwell SFF Edition (24 GB), un Ryzen 9 7945HX3D et le modèle Unsloth MTP UD-IQ3_XXS fixé pour ce projet. Les paramètres utilisés étaient une génération gloutonne, MTP 3, les capacités de contexte indiquées ci-dessous, des blocs de prompt de 512 tokens, le placement automatique des experts et le chemin hybride CUDA Graph optionnel activé.

**Des mesures sur des machines équipées de GPU physiques de 8 GB, 12 GB et 16 GB sont prévues ultérieurement.** Les lignes à mémoire réduite du tableau ci-dessous ont été mesurées en limitant la mémoire disponible sur le même GPU de 24 GB et ne sont pas des benchmarks de GPU physiques de 8/12/16 GB.

| Mémoire disponible | Capacité de contexte | Génération INT8 | Génération FP8 native | Remarques |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 121.28 tok/s | 122.73 tok/s | GPU entièrement disponible ; tous les experts résidents |
| environ 16 GB | 256K | 96.98 tok/s | 112.39 tok/s | 8 GiB occupés séparément |
| environ 12 GB | 128K | 104.28 tok/s | 107.61 tok/s | 12 GiB occupés séparément |
| environ 8 GB | 64K | 81.43 tok/s | 86.69 tok/s | 16 GiB occupés séparément |

Il s’agit de vitesses de décodage agrégées sur trois entrées courtes (34, 46 et 1,599 tokens), chacune produisant deux fois 128 tokens. Le chargement et le traitement des entrées sont exclus. **64K–256K sont des capacités de cache allouées, pas des longueurs d’entrée.** Toutes les lignes à mémoire réduite utilisaient des allocations indépendantes sur le même GPU de 24 GB et effectuaient de véritables calculs d’experts sur le CPU.

Les chiffres pour 24/16 GB proviennent de l’exécution précédente. Les cas ajoutés pour 12/8 GB utilisaient la version actuelle et la sélection automatique du contexte (128K/64K). La capacité de contexte influe également sur la vitesse ; ces lignes présentent donc différents réglages pratiques. L’avantage du FP8 est ici un résultat mesuré dans ces conditions, pas une garantie générale ni une comparaison de qualité.

## Quel projet choisir ?

| Votre objectif et votre matériel | Projet |
| --- | --- |
| Vous souhaitez une inférence Qwen3.6-35B-A3B exclusivement par API sous **Windows**, avec **8–24 GB de VRAM et 32 GB de mémoire système au maximum**, par exemple sur un PC de jeu courant ou un ordinateur portable de jeu. | TinyStrata (**Windows uniquement**, en développement) |
| Vous disposez à la fois **d’un GPU de la classe RTX 3060 ou supérieure** et **de 64 GB de mémoire système ou plus**. | Nous recommandons [Strata (projet amont)](https://github.com/Niko1221/Strata). |
| Vous souhaitez une **interface de chat intégrée**. | Nous recommandons [Strata (projet amont)](https://github.com/Niko1221/Strata). |

**TinyStrata prend uniquement en charge Windows.** Linux et macOS ne font pas partie du périmètre de prise en charge de ce projet. Ces indications de mémoire système servent de repères pour choisir un projet. Les besoins réels en mémoire dépendent du modèle, de la quantification, de la longueur du contexte et de la VRAM disponible.

## Répartition de la mémoire

Un modèle MoE contient de nombreux **experts** : des réseaux neuronaux plus petits sélectionnés pour chaque token. TinyStrata conserve autant d’experts que possible dans la mémoire GPU (VRAM). Pendant la génération, il peut calculer les experts sélectionnés qui ne sont pas en cache sur le CPU, en utilisant la mémoire système. Lors du traitement de lots d’entrée plus volumineux, il peut à la place transférer les poids des experts depuis la RAM via des tampons GPU temporaires. Les experts sélectionnés par le modèle restent inclus dans le calcul.

Les fichiers du modèle téléchargés représentent environ **15.0 GB** au total : **14.1 GB** pour le modèle de langage (poids MTP compris) et **0.9 GB** pour le projecteur de vision (`mmproj-F16.gguf`). Le modèle de langage n’a pas besoin de tenir entièrement dans la VRAM. La mémoire GPU doit également accueillir l’encodeur de vision et les tampons de traitement d’image, le **cache KV** de la conversation, les autres poids du modèle et les tampons de calcul temporaires :

![Répartition de la mémoire entre la VRAM du GPU et la mémoire système dans TinyStrata](assets/memory-sharing.png)

Il s’agit d’un schéma de répartition, pas d’un partitionnement de taille fixe. À chaque chargement du modèle, TinyStrata démarre et préchauffe l’encodeur de vision sur le GPU avant de charger le modèle de langage. Il alloue ensuite le cache KV, les autres poids et les tampons de travail, puis adapte le cache d’experts à la VRAM libre restante tout en conservant une marge. Sous Windows, il vérifie à la fois la mémoire libre indiquée par CUDA et celle indiquée par le pilote NVIDIA pour l’ensemble du périphérique, et utilise la plus petite valeur disponible si les deux requêtes aboutissent. Si l’allocation échoue ou laisse une marge trop faible, il réessaie avec moins d’experts.

Le contenu du cache change selon les experts utilisés ; sa capacité est recalculée au rechargement du modèle. Les autres applications ouvertes par la suite peuvent toujours consommer de la VRAM. La réserve constitue une marge lors du placement des experts, pas une garantie permanente que 768 MiB resteront libres.

Davantage d’experts résidents sur le GPU signifie généralement moins de travail pour le CPU. En mode hybride, la mémoire système conserve également des copies des experts mis en cache dans la VRAM ; ajouter de la VRAM ne réduit donc pas l’utilisation de la RAM octet pour octet. Laissez aussi de la RAM au système d’exploitation et aux autres applications. Une plus grande **capacité de RAM** permet de conserver les données du modèle, mais la vitesse dépend également du CPU, de la bande passante de la RAM et du GPU. La « mémoire GPU partagée » de Windows diffère du calcul délibéré des experts sur le CPU par TinyStrata.

Le cache KV reste dans la VRAM ; la mémoire système contient les données des experts pour ce mécanisme de déport. Une longue conversation a donc toujours besoin de suffisamment de mémoire GPU pour son cache KV.

## Paramètres automatiques

Les nouvelles configurations choisissent automatiquement KV/Attention et la capacité de contexte. Vous n’avez pas besoin de sélectionner un profil pour 8 GB, 12 GB ou 16 GB.

![Paramètres matériels automatiques pour Attention, le cache KV et la capacité de contexte](assets/automatic-settings.png)

Une RTX 3060 sélectionne INT8. Les environnements RTX 40/50 pris en charge peuvent sélectionner le FP8 natif ; le moteur vérifie les noyaux compilés et les capacités du GPU ainsi que la génération matérielle. Si le FP8 natif n’est pas disponible, le mode automatique utilise INT8. Cela sélectionne un chemin pris en charge sans exécuter de benchmark de vitesse ou de qualité au démarrage. **L’Attention FP8 native reste expérimentale.**

La capacité de contexte est l’espace maximal réservé à l’entrée et à la sortie générée réunies. Une capacité plus élevée utilise davantage de mémoire KV même avec une conversation courte, ce qui laisse moins de VRAM pour les experts. La capacité n’implique pas non plus une vitesse identique pour toutes les longueurs de conversation : traiter effectivement un long document demande davantage de travail. À titre indicatif, le cache KV INT8 seul occupe environ **0.71 GiB à 64K, 1.42 GiB à 128K et 2.84 GiB à 256K**, couche MTP comprise ; les autres poids et tampons de travail s’y ajoutent. Dans cette implémentation, le FP8 natif utilise approximativement le même espace de stockage KV que l’INT8 ; son intérêt réside dans un chemin de calcul différent, plutôt que dans un cache nettement plus petit.

La VRAM libre désigne la mémoire disponible au moment du chargement, pas le chiffre imprimé sur la boîte du GPU. Les autres applications peuvent donc entraîner la sélection d’un contexte plus petit. Si un rechargement réduit la capacité, l’API rejette une entrée qui ne tient plus au lieu de la tronquer silencieusement.

La quantification des poids GGUF et la précision KV sont distinctes. Le modèle UD-IQ3_XXS sélectionné possède des poids compressés ; INT8 ou FP8 décrit le cache de conversation et son chemin Attention. Une précision KV plus faible peut modifier les réponses ; ces modes ne promettent donc pas une sortie identique à celle du cache KV FP32. L’implémentation conserve toutes les positions de contexte visibles et chaque expert sélectionné par le modèle.

Les nouvelles configurations du lanceur utilisent MTP 3, qui propose plusieurs tokens à l’avance et les vérifie avec le modèle principal. Cela peut accélérer la génération, mais le gain dépend du nombre de propositions acceptées. La vérification vise à préserver le comportement de décodage du modèle principal ; des différences dans les chemins d’exécution numérique peuvent néanmoins modifier les tokens. Activer MTP ne garantit donc pas une sortie identique à celle obtenue en le désactivant.

MTP prend en charge la génération gloutonne et l’échantillonnage fondé sur la température, y compris les pénalités de répétition, de fréquence et de présence. Pour les requêtes avec échantillonnage, le modèle principal échantillonne chaque ligne de vérification, et seul le préfixe concordant de la proposition est accepté.

Les paramètres explicites remplacent les choix automatiques, et les paramètres déjà sélectionnés manuellement restent en vigueur. Par exemple, `run-api.bat --kv-type int8` impose INT8 ; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` rétablit la sélection automatique de KV, du contexte et du placement des experts.

## Serveur API

**TinyStrata fonctionne exclusivement par API.** Il fournit des API Chat Completions (`/v1/chat/completions`) et Responses (`/v1/responses`) compatibles avec OpenAI ainsi qu’une API Messages (`/v1/messages`) compatible avec Anthropic, avec génération en streaming. Connectez un client API ou une interface externe ; TinyStrata ne comprend pas d’interface de chat intégrée.

Par défaut, le raisonnement est activé, avec `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` et un maximum de **32,768 tokens de sortie**, raisonnement compris. Les paramètres définis pour chaque requête sont prioritaires.

L’API écoute par défaut sur `http://127.0.0.1:18080`. Pour utiliser un autre port, exécutez `run-api.bat --port 18081` depuis le dossier d’installation de TinyStrata.

L’API peut démarrer avant l’installation des fichiers du modèle. `POST /v1/model/download` avec un corps JSON vide (`{}`) télécharge le [modèle de langage Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) fixé pour ce projet (14.1 GB, poids MTP compris) et son [projecteur de vision](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), soit environ **15.0 GB au total**. Il vérifie les sommes de contrôle SHA-256 des deux fichiers, extrait le tokenizer et les charge. `GET /v1/model/download` indique la progression ; les téléchargements interrompus peuvent reprendre. Une nouvelle requête de téléchargement réutilise la tâche active, et `POST /v1/model/download/cancel` peut annuler le transfert ou la vérification. La fin du téléchargement ne signifie pas que l’inférence est prête : attendez que l’état du téléchargement passe à `ready` et que `/health` indique `loaded: true` et `images: true`.

Le programme d’installation prépare le moteur et les bibliothèques d’exécution GPU, puis télécharge automatiquement les deux fichiers du modèle. Les configurations existantes conservent le modèle local sélectionné jusqu’à ce que vous le changiez explicitement.

## Remerciements

TinyStrata est un fork indépendant de [Strata](https://github.com/Niko1221/Strata) et s’appuie largement sur son implémentation. Merci à [Niko](https://github.com/Niko1221) et à la communauté Strata pour le moteur et le travail qui ont rendu TinyStrata possible.

Merci également à [Aaron Flin](https://github.com/aflin) et au [projet MoMoE](https://github.com/aflin/momoe) pour le code du cache KV, d’attention et des experts quantifiés adapté dans TinyStrata.

## Licence

MIT. Voir [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Attribution au projet amont : [NOTICE](NOTICE). GGML : [licence MIT](third_party/ggml/LICENSE).
