# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

**O TinyStrata é um mecanismo de inferência de alta velocidade em desenvolvimento para modelos MoE da classe 30B, exclusivo para Windows e acessível apenas por API.** Ele é voltado principalmente a PCs equipados com placas de vídeo comuns de **8–24 GB** de VRAM e com **32 GB de RAM do sistema ou menos**.

**Há suporte a GPUs NVIDIA (CUDA). A rota experimental AMD HIP/ROCm usa KV INT8 e seu funcionamento foi verificado por um usuário da comunidade em uma Radeon RX 9070 XT.**

## Por que desenvolvemos o TinyStrata?

Com o aumento dos preços dos componentes, montar um PC de ponta está cada vez mais fora do alcance de muitas pessoas.

Por isso, o TinyStrata foi criado para executar IA localmente em alta velocidade em PCs equipados com placas de vídeo comuns da classe RTX 3060 ou superior. Nosso foco é fazer modelos MoE relativamente pequenos, da classe 30B, rodarem o mais rápido possível.

Nosso principal público são PCs com **8–24 GB de VRAM e 32 GB de RAM do sistema ou menos**. Acreditamos que essa faixa representa as configurações mais frequentes de PCs equipados com placas de vídeo comuns.

Baseado no Strata, o TinyStrata busca usar a memória da GPU de forma eficiente para executar modelos MoE da classe 30B com rapidez, mesmo com VRAM limitada. Nosso objetivo é permitir que você trabalhe com conversas e documentos longos no seu próprio PC.

O desenvolvimento se concentra na inferência local no Windows e no acesso a ela por meio de uma API.

O TinyStrata escolhe automaticamente as configurações de inferência e memória para o seu hardware, para que você possa começar a usar a API sem configurações complicadas nem ajustes manuais.

## Por que não há uma interface incluída?

Já existem muitas interfaces e ambientes de agentes compatíveis com APIs para LLMs. Como o TinyStrata também oferece suporte a formatos comuns de API, você pode conectar e usar os que preferir. Se configurar a conexão parecer trabalhoso, você pode pedir a um agente de IA que faça isso. Ao deixar a interface a cargo do software existente, nos concentramos no desenvolvimento do mecanismo de inferência e da API.

## Foco de modelos

O [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) continua sendo o modelo padrão pelo equilíbrio entre velocidade, consumo de memória e qualidade das respostas em PCs com placas de vídeo comuns. O TinyStrata se concentra em modelos MoE pequenos e também oferece **suporte ao Gemma 4 26B-A4B-it e ao LFM2.5-24B-A2B**.

O **Qwen3.6-35B-A3B UD-IQ3_XXS** usa o [GGUF MTP fixado da Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf). Os dois arquivos somam aproximadamente **15.0 GB**: modelo de linguagem, incluindo os pesos MTP, de **14.1 GB** e projetor de visão (`mmproj-F16.gguf`) de **0.9 GB**.

O **Gemma 4 26B-A4B-it QAT Q4_0** usa [o modelo e o projetor de visão oficiais do Google](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) com [o assistente MTP QAT correspondente da Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Os três arquivos somam **15.886 GB**: modelo **14.439 GB**, projetor **1.195 GB** e assistente **0.252 GB**.

O **LFM2.5-24B-A2B Q4_K_M** usa o [GGUF oficial da LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF). Seu único arquivo de modelo de linguagem ocupa **14.53 GB**.

Os três motores usam cache automático de especialistas na VRAM, cálculo de especialistas na CPU a partir da RAM, streaming de pesos durante a entrada e CUDA Graphs. Compartilham a mesma API e permitem trocar o modelo sem reiniciar, com configurações salvas separadamente para cada modelo. Os especialistas selecionados não são omitidos. Também oferecem crescimento automático da memória KV física, sem alterar a capacidade máxima de contexto.

| Recurso | [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B-it) | [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| --- | --- | --- | --- |
| Contexto máximo | 256K | 256K | 128K |
| Entrada de imagens | Sim; codificador na GPU por padrão | Sim; codificador na GPU por padrão | Não |
| MTP | 3 | 3 | Não |
| Formatos KV | FP32 / INT8 / FP8 nativo | FP16 / INT8 / FP8 nativo (experimental) | FP16 / INT8 / FP8 nativo |
| KV automático (padrão) | FP8 → INT8 | INT8 | FP8 → INT8 |

Os cabeçalhos apontam para as fichas oficiais dos modelos. **FP8 → INT8** indica KV FP8 e Attention nativa com Tensor Cores FP8 quando a GPU e os kernels compilados permitem, ou INT8 caso contrário. A precisão KV independe da quantização dos pesos GGUF; KV de menor precisão pode alterar a saída em relação a FP32 (Qwen) ou FP16 (Gemma/LFM).

O Gemma usa INT8 por padrão. O FP8 nativo é experimental e pode ser selecionado explicitamente com `run-api.bat --model-family gemma4 --kv-type fp8-mma`.

A **capacidade de contexto automática** usa a VRAM total da GPU, como mostrado abaixo; a VRAM livre determina o tamanho do cache de especialistas. Um valor explícito de `--max-context N` tem prioridade.

| VRAM total (exemplos) | Qwen3.6 / Gemma 4 | LFM2.5 |
| --- | ---: | ---: |
| 8 GB | 64K | 64K |
| 12 GB | 128K | 128K |
| 16 GB ou mais | 256K | 128K |

As faixas de capacidade consideram diferenças nos valores informados pelo driver, portanto uma placa de 16 GB que informe 15.x GiB ainda seleciona 256K para Qwen e Gemma.

Se o executável de um modelo adicional não estiver disponível, veja as [instruções de compilação manual](README.md#model-focus).

## Como instalar

Execute no PowerShell de 64 bits para instalar o TinyStrata e iniciar a API em `http://127.0.0.1:18080/v1`:

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Para iniciar novamente após reiniciar o Windows, execute este comando ou dê um clique duplo no mesmo arquivo `run-api.bat`:

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

Você também pode passar o link do [repositório do TinyStrata no GitHub](https://github.com/kbc20f11/TinyStrata) a um agente de IA e deixar a instalação por conta dele.

### Trocar de modelo sem reiniciar

O instalador inicia o Qwen por padrão. Para preparar o Gemma no mesmo servidor API, inicie o download e consulte o progresso:

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

Repita a consulta até `state` ser `ready` e então solicite a troca explicitamente. O download de um modelo inativo apenas o prepara, sem ativá-lo:

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

Para iniciar diretamente com o Gemma, execute `run-api.bat --model-family gemma4`; a API usa `http://127.0.0.1:18082/v1`. Para o LFM, substitua `gemma-4-26b-a4b-it` por `lfm2.5-24b-a2b` nos pedidos de download, progresso e troca acima, ou execute `run-api.bat --model-family lfm2` para iniciá-lo diretamente em `http://127.0.0.1:18081/v1`.

Consulte `GET /v1/model/switch` até `state` ser `ready`. Para voltar ao Qwen, envie o mesmo pedido com `{"model":"qwen3.6-35b-a3b"}`. Iniciar um download ou uma troca retorna `202`; uma troca durante uma geração retorna `409`, então tente novamente quando ela terminar. Pedidos de inferência durante a troca recebem `503`. A porta e a autenticação não mudam. A seleção vale para a sessão atual do servidor; ao reiniciar, é usado o modelo escolhido no inicializador.

## Velocidade medida — Qwen3.6

Medido em **2026-10-21** no Windows com uma RTX PRO 4000 Blackwell SFF Edition (24 GB), um Ryzen 9 7945HX3D, 64 GB de RAM e o modelo fixo Unsloth MTP UD-IQ3_XXS. O codificador de imagens permaneceu na GPU durante as solicitações de texto pela API. Foram ativados o raciocínio, os padrões de amostragem do modelo (temperatura 0.7) e MTP 3. Os blocos de entrada e a distribuição de especialistas foram automáticos; o contexto foi fixado conforme a classe de VRAM. O caminho híbrido de CUDA Graph permaneceu desativado por padrão e os grafos habituais de execução inteiramente na GPU continuaram ativos.

| Condição de memória disponível | Capacidade de contexto | Geração com INT8 | Geração com FP8 nativo | Notas |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 117.80 tok/s | 131.34 tok/s | Todos os pesos residentes na VRAM |
| Aprox. 16 GB | 256K | 98.90 tok/s | 101.68 tok/s | 8 GiB reservados; cálculo de especialistas na CPU e na GPU |
| Aprox. 12 GB | 128K | 87.99 tok/s | 86.66 tok/s | 12 GiB reservados; cálculo de especialistas na CPU e na GPU |
| Aprox. 8 GB | 64K | 60.69 tok/s | 60.69 tok/s | FP8: sem cache de Experts na GPU |

Os mesmos três textos foram executados duas vezes por configuração FP8/INT8 com os padrões de amostragem do modelo, gerando 128 tokens por solicitação, incluindo o raciocínio. A velocidade é o total de tokens dividido pelo tempo total de decodificação, sem carregamento nem processamento da entrada; o contexto indica capacidade de cache. As linhas com menos memória reservam 8, 12 ou 16 GiB na mesma GPU de 24 GB e usam o contexto dessa classe de VRAM; não são testes de GPUs físicas de 16/12/8 GB.

## Velocidade medida — Gemma 4

Medido no mesmo PC Windows com 64 GB de RAM, Gemma 4 26B-A4B-it QAT Q4_0 e o assistente MTP QAT correspondente. O codificador de imagens permaneceu na GPU durante as solicitações de texto pela API. MTP 3, CUDA Graphs e os padrões de raciocínio e amostragem do modelo estavam ativados; o contexto foi fixado conforme a classe de VRAM, enquanto os blocos de entrada e a distribuição de especialistas foram escolhidos automaticamente.

FP8 nativo e INT8 foram medidos em **2026-10-21**.

| VRAM disponível | Capacidade de contexto | INT8 (padrão) | FP8 nativo (experimental) | Observações |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 113.20 tok/s | 112.31 tok/s | INT8 / FP8: Cálculo de especialistas na GPU |
| Aprox. 16 GB | 256K | 100.08 tok/s | 96.67 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |
| Aprox. 12 GB | 128K | 81.66 tok/s | 82.08 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |
| Aprox. 8 GB | 64K | 51.54 tok/s | 53.41 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |

Os mesmos três textos foram executados duas vezes por configuração FP8/INT8 com os padrões de amostragem do modelo, gerando 128 tokens por solicitação, incluindo o raciocínio. A velocidade é o total de tokens dividido pelo tempo total de decodificação, sem carregamento nem processamento da entrada; o contexto indica capacidade de cache. As linhas com menos memória reservam 8, 12 ou 16 GiB na mesma GPU de 24 GB e usam o contexto dessa classe de VRAM; não são testes de GPUs físicas de 16/12/8 GB.

## Velocidade medida — LFM2.5

Medido no mesmo PC Windows com 64 GB de RAM e o LFM2.5-24B-A2B Q4_K_M oficial, sem MTP ou Vision. CUDA Graphs e os padrões de raciocínio e amostragem do modelo estavam ativados; o contexto foi fixado conforme a classe de VRAM, enquanto os blocos de entrada e a distribuição de especialistas foram escolhidos automaticamente.

FP8 nativo e INT8 foram medidos em **2026-10-21**.

| VRAM disponível | Capacidade de contexto | INT8 | FP8 nativo (padrão) | Observações |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 128K | 140.58 tok/s | 141.80 tok/s | INT8 / FP8: Cálculo de especialistas na GPU |
| Aprox. 16 GB | 128K | 135.63 tok/s | 138.21 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |
| Aprox. 12 GB | 128K | 110.11 tok/s | 114.72 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |
| Aprox. 8 GB | 64K | 71.10 tok/s | 73.92 tok/s | INT8 / FP8: Cálculo de especialistas na GPU e na CPU |

Os mesmos três textos foram executados duas vezes por configuração FP8/INT8 com os padrões de amostragem do modelo, gerando 128 tokens por solicitação, incluindo o raciocínio. A velocidade é o total de tokens dividido pelo tempo total de decodificação, sem carregamento nem processamento da entrada; o contexto indica capacidade de cache. As linhas com menos memória reservam 8, 12 ou 16 GiB na mesma GPU de 24 GB e usam o contexto dessa classe de VRAM; não são testes de GPUs físicas de 16/12/8 GB.

## Velocidade medida — AMD Radeon RX 9070 XT (HIP, experimental)

Medido em **2026-10-24** em um PC com Windows 11 e uma Radeon RX 9070 XT (gfx1201, 16 GB, que também controla a tela), 94 GB de RAM e HIP SDK 7.2, com os mesmos arquivos Qwen3.6 UD-IQ3_XXS, Gemma 4 QAT Q4_0 (com o assistente MTP) e LFM2.5 Q4_K_M de cima. Tudo ficou nos padrões por meio de `run-api.bat`: o backend HIP adiciona `--kv-whole`, e a capacidade de contexto, o MTP e o codificador de imagens na GPU foram escolhidos automaticamente (o LFM2.5 não tem MTP nem Vision). O raciocínio estava ativado e a amostragem ficou nos padrões do modelo, exceto que cada solicitação usou `seed` 1234.

| Modelo | Capacidade de contexto | Decodificação, primeiras 3 solicitações após a inicialização (duas inicializações) | Decodificação, estável (mediana de 21 solicitações) | Entrada de cerca de 2.7K tokens, primeira / seguintes |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 256K | 68.3 / 71.4 tok/s | 70.9 tok/s (62.2 a 78) | 514 e 485 / cerca de 1200 tok/s |
| Gemma 4 QAT Q4_0 | 256K | 36.7 / 37.7 tok/s | 61.7 tok/s (50.6 a 71.9) | 1518 e 1849 / cerca de 1660 tok/s |
| LFM2.5 Q4_K_M | 128K | 80.9 / 86.2 tok/s | 101.1 tok/s (94.8 a 110.6) | 2135 e 2214 / cerca de 2410 tok/s |

Foram usados os mesmos três textos de cima, gerando 128 tokens, incluindo o raciocínio. A velocidade é o número de tokens gerados dividido pelo tempo de decodificação de `/v1/status`, sem carregamento nem processamento da entrada. «Primeiras 3 solicitações» são os três textos em ordem fixa logo depois de a API informar que o modelo estava carregado, em média, em duas inicializações separadas. «Estável» são as rodadas 4 a 10 de dez rodadas dos três textos em uma inicialização (21 solicitações), com a faixa entre parênteses. As linhas de entrada usam um texto de 2669 a 2676 tokens, enviado a cada vez com uma primeira linha diferente para que nada estivesse em cache: «primeira» é a primeira solicitação longa de cada inicialização e «seguintes» as três seguintes. O pico de VRAM do adaptador inteiro foi de 15.55 a 15.64 GiB de cerca de 15.92 GiB, incluindo cerca de 2.4 GiB usados por outros programas.

- Nesta placa, o motor do Qwen falha com o KV crescente padrão (veja a issue #1); com `--kv-whole`, os três modelos rodaram sem erros e sem dumps do watchdog de GPU do Windows. Gemma 4 e LFM2.5 só foram executados nesta configuração.
- O Qwen3.6 decodifica nas primeiras solicitações com a mesma velocidade do regime estável, dentro da variação entre execuções, mas sua primeira entrada longa é cerca de 2.4 vezes mais lenta, nas duas inicializações. Gemma 4 e LFM2.5 decodificam mais devagar nas primeiras solicitações (cerca de 37 contra 62 tok/s, e 81 a 86 contra 101 tok/s), nas duas inicializações. Os logs do motor mostram muita troca no cache de especialistas durante a primeira solicitação (745 trocas no Gemma 4, contra 164 a 372 depois) e um tamanho de VRAM que diminui ao longo das primeiras solicitações, de 8.5 para 4.7 GiB no Gemma 4 e de 11.6 para 10.1 GiB no LFM2.5. Não descobri por que ele diminui nem se isso explica a partida lenta.
- Com a semente fixa, as saídas das três primeiras solicitações foram idênticas nas duas inicializações. Repetir o mesmo texto depois, na mesma sessão, dá saídas diferentes, então as linhas estáveis misturam diferenças de saída com velocidade.
- Cada solicitação levou de 1 a 4 s e solicitações isoladas variam cerca de 10%: não interprete diferenças de poucos tok/s. É uma máquina e uma sessão. Esta placa tem 16 GB e as tabelas acima usam uma placa de 24 GB, portanto as linhas não são comparáveis.

## Velocidade medida — NVIDIA GeForce RTX 5060 (8 GB)

Medido em **2026-10-25** em um PC com Windows 11 e uma GeForce RTX 5060 (8 GB, driver 616.92), um Core i7-10700K (8 núcleos), 32 GB de DDR4-3200 e a GPU em um link PCIe 3.0 x8, com os mesmos arquivos Qwen3.6 UD-IQ3_XXS, Gemma 4 QAT Q4_0 (com o assistente MTP) e LFM2.5 Q4_K_M de cima. Tudo ficou nos padrões por meio de `run-api.bat` (backend CUDA; capacidade de contexto, tipo de KV, bloco de entrada, posicionamento dos especialistas e MTP escolhidos automaticamente; o LFM2.5 não tem MTP nem Vision). O raciocínio estava ativado e a amostragem ficou nos padrões do modelo, exceto que todas as solicitações usaram `seed` 1234.

| Modelo | Capacidade de contexto | Decodificação, primeiras 3 solicitações após a inicialização (duas inicializações) | Decodificação, estável (mediana de 21 solicitações) | Entrada de cerca de 2.5K tokens, primeira / seguintes |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 64K | 27.2 / 33.7 tok/s | 29.2 tok/s (24.1 a 35.0) | 61 e 62 / cerca de 104 (89 a 118) tok/s |
| Gemma 4 QAT Q4_0 | 64K | 24.0 / 25.6 tok/s | 25.7 tok/s (22.2 a 28.7) | 333 e 369 / cerca de 364 (361 a 371) tok/s |
| LFM2.5 Q4_K_M | 64K | 43.1 / 43.3 tok/s | 53.8 tok/s (42.4 a 56.8) | 595 e 685 / cerca de 716 (690 a 762) tok/s |

Cada solicitação gerou 128 tokens, incluindo o raciocínio. A velocidade é o número de tokens gerados dividido por `predicted_ms` de `/v1/status`, sem carregamento nem processamento da entrada. «Primeiras 3 solicitações» são três textos de entrada em ordem fixa logo depois de a API informar que o modelo estava carregado, em duas inicializações separadas (cada valor é o total de tokens dividido pelo tempo total das três). «Estável» são as rodadas 4 a 10 de dez rodadas dos três textos na primeira inicialização (21 solicitações), com a faixa entre parênteses. As linhas de entrada usam uma entrada de 2440 (Qwen3.6), 2446 (Gemma 4) ou 2545 (LFM2.5) tokens, enviada a cada vez com uma primeira linha diferente para que nada ficasse em cache: «primeira» é a primeira solicitação longa de cada inicialização e «seguintes» as três seguintes das duas inicializações. Os três textos curtos e a entrada longa foram escritos para esta medição e não são os das seções acima, portanto as linhas não são diretamente comparáveis com elas. A memória de GPU em uso após as execuções foi de 6.5 a 6.9 GiB de cerca de 7.9 GiB, incluindo a área de trabalho.

- O Qwen3.6 manteve 12 (primeira inicialização) e 22 (segunda inicialização) dos 256 especialistas por camada na VRAM e os outros 11.4 GiB na RAM do sistema fixada, com um bloco de entrada de 512. As duas inicializações tinham 4.6 e 5.3 GiB de VRAM livre, e as velocidades posteriores de entrada foram de cerca de 90 e 117 tok/s, o que combina com mais especialistas residentes serem mais rápidos; não testei isso separadamente.
- As velocidades de entrada são baixas em comparação com a placa de 16 GB de cima. As razões prováveis são que a maioria dos especialistas não está na VRAM, que os especialistas fora da VRAM são calculados em 8 threads de CPU (Qwen3.6) ou transferidos para a GPU por um link PCIe 3.0 x8 de cerca de 8 GB/s (Gemma 4 e LFM2.5, cujos logs mostram especialistas em etapa intermediária), e que a VRAM livre é pequena demais para escolher um bloco maior. Outros valores de `--chunk` não foram testados, então isto não está confirmado.
- Solicitações isoladas variam 10% ou mais e as duas inicializações diferem em até 6 tok/s: não interprete diferenças pequenas. É uma máquina e uma sessão.

## Qual projeto devo usar?

| Seu objetivo e seu hardware | Projeto |
| --- | --- |
| Você quer inferência do Qwen3.6 / Gemma 4 / LFM2.5 apenas por API no **Windows**, com **8–24 GB de VRAM e 32 GB de RAM do sistema ou menos**, como em um PC equipado com uma placa de vídeo comum. | TinyStrata (**somente Windows**, em desenvolvimento) |
| Você tem tanto **uma GPU da classe RTX 3060 ou superior** quanto **64 GB de RAM do sistema ou mais**. | Recomendamos o [Strata (projeto original)](https://github.com/Niko1221/Strata). |
| Você quer uma **interface de chat incluída**. | Recomendamos o [Strata (projeto original)](https://github.com/Niko1221/Strata). |

## Como a memória é compartilhada — Qwen3.6

Um modelo MoE contém muitos **especialistas**: redes neurais menores selecionadas para cada token. O TinyStrata mantém o maior número possível de especialistas na memória da GPU (VRAM). Durante a geração, ele pode calcular na CPU os especialistas selecionados que não estão em cache, usando a RAM do sistema. Ao processar lotes de entrada maiores, ele pode, em vez disso, transferir os pesos dos especialistas da RAM por meio de buffers temporários da GPU. Todos os especialistas selecionados pelo modelo participam do cálculo.

Os arquivos de modelo baixados somam cerca de **15.0 GB**: **14.1 GB** para o modelo de linguagem (incluindo os pesos de MTP) e **0.9 GB** para o projetor de visão (`mmproj-F16.gguf`). O modelo de linguagem não precisa caber inteiro na VRAM. A memória da GPU também precisa de espaço para o codificador de visão e os buffers de processamento de imagens, o **cache KV** da conversa, os demais pesos do modelo e os buffers temporários de cálculo:

![Como o TinyStrata compartilha a memória entre a VRAM da GPU e a RAM do sistema](assets/memory-sharing.png)

Este é um diagrama de distribuição no modo padrão de imagens na GPU, não uma partição de tamanho fixo. A cada carregamento do modelo, o TinyStrata inicia e aquece o codificador de visão da GPU antes de carregar o modelo de linguagem; seus pesos e buffers de trabalho aquecidos permanecem residentes. Em seguida, aloca as páginas KV iniciais, os demais pesos e os buffers de trabalho, e dimensiona o cache de especialistas conforme a VRAM livre restante. No Windows, usa o menor valor entre a memória livre informada pelo CUDA e a VRAM dedicada livre informada pelo driver NVIDIA, quando ambos estão disponíveis.

O conteúdo do cache muda conforme os especialistas usados. Com o crescimento KV padrão, conversas curtas deixam mais VRAM para especialistas; conforme a sequência cresce, o cache de especialistas devolve memória ao KV. Novas solicitações mais curtas podem devolver páginas KV não utilizadas ao cache de especialistas. `run-api.bat --kv-whole` aloca toda a capacidade KV antecipadamente; `run-api.bat --kv-grow` restaura o crescimento, exceto no backend AMD HIP, que sempre usa `--kv-whole` e rejeita `--kv-grow` (veja a issue #1). Outros aplicativos ainda podem consumir VRAM.

A alocação de especialistas normalmente visa uma margem de 768 MiB. No modo `--kv-whole` do Qwen, como no Strata, quando falta pouco para todos os especialistas do modelo principal caberem na VRAM, ela pode liberar as cópias empacotadas dos pesos Q8, reduzir um buffer de pontuações de Attention quando existente e diminuir a margem para 256 MiB; os especialistas MTP podem executar na CPU. `--vram-reserve-mib N` fixa a reserva explicitamente. Esse é um objetivo na alocação, não uma garantia permanente de memória livre.

Mais especialistas residentes na GPU geralmente significam menos trabalho para a CPU. No modo híbrido, a RAM do sistema também mantém cópias de apoio dos especialistas armazenados em cache na VRAM, portanto adicionar VRAM não reduz o uso de RAM byte por byte. Deixe RAM também para o sistema operacional e outros aplicativos. Uma maior **capacidade de RAM** permite manter os dados do modelo, mas a velocidade também depende da CPU, da largura de banda da RAM e da GPU. A "memória de GPU compartilhada" do Windows é diferente do cálculo deliberado de especialistas na CPU feito pelo TinyStrata.

O cache KV permanece na VRAM; a RAM do sistema armazena os dados dos especialistas para esse mecanismo de transferência de trabalho. Portanto, uma conversa longa ainda precisa de memória de GPU suficiente para seu cache KV.

## Configurações automáticas — Qwen3.6

Novas configurações escolhem KV/Attention e a capacidade de contexto automaticamente. Você não precisa selecionar uma receita para 8 GB, 12 GB ou 16 GB.

No Qwen3.6, uma RTX 3060 seleciona INT8. Ambientes compatíveis com RTX das séries 40/50 podem selecionar FP8 nativo; o mecanismo verifica os kernels compilados e os recursos da GPU, além da geração do hardware. Se o FP8 nativo não estiver disponível, o modo automático usa INT8. Isso escolhe um caminho compatível sem executar um teste de velocidade ou qualidade na inicialização.

O mecanismo nativo estima o tamanho dos blocos de entrada na inicialização, escolhendo 512, 1024, 2048 ou 4096 tokens conforme a VRAM livre, a capacidade de contexto e se os especialistas cabem na GPU. Blocos maiores podem reduzir transferências repetidas de pesos da RAM para a GPU durante o processamento da entrada. `run-api.bat --chunk N` fixa o tamanho; `run-api.bat --chunk auto` restaura a seleção automática nativa.

A capacidade de contexto é o limite lógico para a entrada e a saída gerada em conjunto. Por padrão, a memória KV física é alocada conforme a conversa cresce, em vez de reservar toda a capacidade para uma conversa curta. Documentos longos continuam exigindo mais cálculo e, por fim, mais memória KV.

Na capacidade máxima, o KV em INT8 do Qwen ocupa cerca de **0.71 GiB em 64K, 1.42 GiB em 128K e 2.84 GiB em 256K**, incluindo a camada MTP; os demais pesos e buffers de trabalho são adicionais. Nesta implementação, o FP8 nativo usa aproximadamente o mesmo armazenamento KV que o INT8; seu benefício é um caminho de cálculo diferente, em vez de um cache substancialmente menor.

A capacidade de contexto é escolhida com base na VRAM total da GPU; o cache de especialistas é então ajustado à VRAM livre restante no carregamento. Outros aplicativos que usam VRAM deixam espaço para menos especialistas em cache na GPU. A API rejeita entradas que excedem a capacidade de contexto, em vez de truncá-las silenciosamente.

A quantização dos pesos GGUF e a precisão do KV são independentes. O modelo UD-IQ3_XXS selecionado tem pesos comprimidos; INT8 ou FP8 descreve o cache da conversa e seu caminho de Attention. O KV de menor precisão pode alterar as respostas, portanto esses modos não prometem uma saída idêntica à do KV em FP32. A implementação mantém todas as posições visíveis do contexto e todos os especialistas selecionados pelo modelo.

Novas configurações do inicializador usam MTP 3, que propõe vários tokens adiante e os verifica com o modelo principal. Isso pode melhorar a velocidade de geração, mas o ganho depende de quantas propostas são aceitas. A verificação busca preservar o comportamento de decodificação do modelo principal; diferenças nos caminhos de execução numérica ainda podem alterar os tokens, portanto ativar MTP não promete uma saída idêntica à obtida ao desativá-lo.

O MTP oferece suporte tanto à geração gulosa quanto à amostragem baseada em temperatura, incluindo penalidades de repetição, frequência e presença. Nas solicitações com amostragem, o modelo principal faz a amostragem de cada linha de verificação, e apenas o prefixo do rascunho que coincide é aceito.

Configurações explícitas substituem as escolhas automáticas, e as configurações existentes selecionadas manualmente continuam em vigor. Por exemplo, `run-api.bat --kv-type int8` fixa INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restaura a seleção automática de KV, contexto e alocação de especialistas.

## Servidor de API

**O TinyStrata funciona apenas por API.** Ele fornece APIs Chat Completions (`/v1/chat/completions`) e Responses (`/v1/responses`) compatíveis com OpenAI e uma API Messages (`/v1/messages`) compatível com Anthropic, incluindo geração em streaming. Conecte um cliente de API ou uma interface externa; o TinyStrata não inclui uma interface de chat.

Configurações novas usam padrões de amostragem próprios de cada modelo. Os valores de cada solicitação têm prioridade; configurações personalizadas salvas são preservadas.

| Configuração | Qwen3.6 | Gemma 4 | LFM2.5 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.7 | 1.0 | 0.1 |
| `top_p` | 0.95 | 0.95 | 1.0 |
| `top_k` | 20 | 64 | 50 |
| `min_p` | 0.0 | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.05 |
| `enable_thinking` | true | true | true (sempre ativado) |
| `max_tokens` (incluindo o raciocínio) | 32768 | 32768 | 32768 |

Por padrão, a API escuta em `http://127.0.0.1:18080`. Para usar outra porta, execute `run-api.bat --port 18081` na pasta em que o TinyStrata foi instalado.

Para o Qwen, `run-api.bat --vision gpu` processa imagens na GPU (padrão para novas configurações); `--vision cpu` usa a CPU e `--vision no` desativa imagens. `--vision-tokens N` define o limite de tokens de imagem (padrão: 4096). Essas escolhas são salvas e mantidas nas próximas inicializações quando as opções são omitidas.

A API pode iniciar antes que os arquivos de modelo sejam instalados. Com Qwen selecionado, `POST /v1/model/download` com um corpo JSON vazio (`{}`) baixa o [modelo de linguagem fixo Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, incluindo os pesos de MTP) e seu [projetor de visão](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), cerca de **15.0 GB no total**. Ele verifica as somas de verificação SHA-256 de ambos os arquivos, extrai o tokenizador e os carrega. `GET /v1/model/download` informa o progresso; downloads interrompidos podem ser retomados. Uma solicitação de download repetida reutiliza a tarefa ativa, e `POST /v1/model/download/cancel` pode cancelar a transferência ou a verificação. A conclusão do download não significa que a inferência está pronta: espere o estado do download se tornar `ready` e `/health` informar `loaded: true` e, com imagens habilitadas, também `images: true`.

O instalador prepara os executáveis dos motores e as bibliotecas de execução da GPU e baixa o Qwen e seu projetor de visão por padrão. Configurações existentes mantêm o modelo local selecionado até que você o altere explicitamente.

## Agradecimentos

O TinyStrata é um fork independente do [Strata](https://github.com/Niko1221/Strata) e se apoia amplamente em sua implementação. Agradecemos a [Niko](https://github.com/Niko1221) e à comunidade Strata pelo mecanismo e pelo trabalho que tornaram o TinyStrata possível.

Agradecemos também a [Aaron Flin](https://github.com/aflin) e ao [projeto MoMoE](https://github.com/aflin/momoe) pelo código de cache KV, atenção e especialistas quantizados adaptado no TinyStrata.

## Licença

MIT. Consulte [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Atribuição ao projeto original: [NOTICE](NOTICE). GGML: [licença MIT](third_party/ggml/LICENSE).
