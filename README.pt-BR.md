# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

**O TinyStrata é um mecanismo de inferência de alta velocidade em desenvolvimento para modelos MoE da classe 30B, exclusivo para Windows e acessível apenas por API.** Ele é voltado principalmente a PCs equipados com placas de vídeo comuns de **8–24 GB** de VRAM e com **32 GB de RAM do sistema ou menos**.

## Por que desenvolvemos o TinyStrata?

Com o aumento dos preços dos componentes, montar um PC de ponta está cada vez mais fora do alcance de muitas pessoas.

Por isso, o TinyStrata foi criado para executar IA localmente em alta velocidade em PCs equipados com placas de vídeo comuns da classe RTX 3060 ou superior. Nosso foco é fazer modelos MoE relativamente pequenos, da classe 30B, rodarem o mais rápido possível.

Nosso principal público são PCs com **8–24 GB de VRAM e 32 GB de RAM do sistema ou menos**. Acreditamos que essa faixa representa as configurações mais frequentes de PCs equipados com placas de vídeo comuns.

Baseado no Strata, o TinyStrata busca melhorar a velocidade de inferência do Qwen3.6-35B-A3B em ambientes com VRAM limitada. Queremos usar a memória da GPU de forma eficiente para que você possa trabalhar com conversas e documentos longos no seu próprio PC.

O TinyStrata concentra seu desenvolvimento na execução de inferência local rápida no Windows e no acesso a ela por meio de uma API. Essa orientação guia nossa implementação e nossos testes.

Também buscamos tornar os primeiros passos o mais simples possível. O TinyStrata escolhe automaticamente as configurações de inferência e memória para o seu hardware, para que você possa usar a API sem configurações complicadas nem ajustes manuais.

## Por que não há uma interface incluída?

Já existem muitas interfaces e ambientes de agentes compatíveis com APIs para LLMs. Como o TinyStrata também oferece suporte a formatos comuns de API, você pode conectar e usar os que preferir. Se configurar a conexão parecer trabalhoso, você pode pedir a um agente de IA que faça isso. Ao deixar a interface a cargo do software existente, nos concentramos no desenvolvimento do mecanismo de inferência e da API.

## Foco de modelos

O [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) continua sendo o modelo padrão pelo equilíbrio entre velocidade, consumo de memória e qualidade das respostas em PCs com placas de vídeo comuns. O TinyStrata se concentra em modelos MoE pequenos e também oferece **suporte ao Gemma 4 26B-A4B-it e ao LFM2.5-24B-A2B**.

O **Gemma 4 26B-A4B-it QAT Q4_0** usa [o modelo principal e o projetor de visão oficiais do Google](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) com [o assistente MTP QAT Q4_0 correspondente da Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). Os três arquivos somam **15.886 GB**: modelo 14.439 GB, projetor 1.195 GB e assistente 0.252 GB. O motor oferece codificador de imagens residente na GPU, cache adaptativo de especialistas na GPU, cálculo de especialistas na CPU a partir da RAM e MTP 3 pela mesma API. Os especialistas selecionados não são omitidos. O Gemma aceita **KV FP16 e INT8**; a seleção automática atual é **FP16**. A entrada de imagens, o MTP 3, o uso combinado de RAM e a troca de modelo foram testados com o modelo real.

O LFM usa o [GGUF Q4_K_M oficial da LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (**14.53 GB**), **KV FP16** e até **128K de contexto**, sem MTP ou Vision. Usa cache automático de especialistas na VRAM, cálculo de especialistas na CPU, streaming de pesos durante a entrada e CUDA Graphs; os especialistas selecionados não são omitidos. Os padrões são `temperature=0.1`, `top_k=50` e `repetition_penalty=1.05`, com raciocínio sempre ativado. As configurações de Qwen, Gemma e LFM são salvas separadamente. O motor LFM ainda está em validação e não garante tokens idênticos ao llama.cpp nem entre diferentes distribuições CPU/GPU.

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

As medições a seguir foram feitas em 2026-10-19 no Windows, com uma RTX PRO 4000 Blackwell SFF Edition (24 GB), um Ryzen 9 7945HX3D e o modelo fixo Unsloth MTP UD-IQ3_XXS. O projetor de visão permaneceu carregado e aquecido na GPU; as solicitações continham apenas texto, sem tempo de codificação de imagens. Foram usados os padrões da API: raciocínio ativado, temperatura 0.7, MTP 3, blocos de entrada de 512 tokens e seleção automática de contexto e alocação de especialistas. O caminho híbrido de CUDA Graph estava desativado por padrão; os grafos normais para execução inteiramente na GPU mantiveram a configuração padrão.

As solicitações omitiram os parâmetros de amostragem e a semente. A única alteração aos seus padrões foi `max_tokens=128` como limite do teste, em vez de 32768. A coluna FP8 usa a seleção automática de KV nesta GPU; para a comparação com INT8, apenas o modo KV foi forçado para INT8.

**Planejamos realizar medições em máquinas reais com GPUs de 8 GB, 12 GB e 16 GB no futuro.** As linhas com memória reduzida da tabela abaixo foram medidas limitando a memória disponível na mesma GPU de 24 GB; elas não são testes de desempenho de GPUs físicas de 8/12/16 GB.

| Condição de memória disponível | Capacidade de contexto | Geração com INT8 | Geração com FP8 nativo | Notas |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | Todos os pesos residentes na VRAM |
| Aprox. 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | 8 GiB reservados; cálculo de especialistas na CPU e na GPU |
| Aprox. 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | 12 GiB reservados; cálculo de especialistas na CPU e na GPU |
| Aprox. 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | 16 GiB reservados; cálculo de especialistas na CPU e na GPU |

## Velocidade medida — Gemma 4

Medido em 2026-10-19 no mesmo PC Windows com RTX PRO 4000 Blackwell SFF Edition (24 GB), Ryzen 9 7945HX3D e 64 GB de RAM. Foram usados o Gemma 4 26B-A4B-it QAT Q4_0 do Google e o assistente MTP QAT correspondente da Unsloth, com codificador de imagens residente na GPU, entradas de texto pela API, MTP 3 e CUDA Graphs ativados. Foram mantidos os padrões do Google: raciocínio ativado, `temperature=1.0`, `top_p=0.95`, `top_k=64`. O contexto e a distribuição de especialistas foram escolhidos automaticamente, com blocos de entrada automáticos de 1120 tokens.

| VRAM disponível | Capacidade de contexto | Geração FP16 | Geração INT8 | Observações |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 105.56 tok/s | — | Cálculo de especialistas na GPU e na CPU; INT8 não medido |
| Aproximadamente 8 GB | 64K | 48.22 tok/s | 48.97 tok/s | Cálculo de especialistas na CPU durante a geração |

Três entradas fixas foram executadas uma vez cada, com 128 tokens de saída incluindo o raciocínio. A velocidade é o total de tokens dividido pelo tempo total de decodificação, sem carregamento nem processamento de entrada. A condição de 8 GB foi obtida reservando 16 GiB separadamente na mesma GPU de 24 GB; não é um teste de uma GPU física de 8 GB. 64K/256K indica a capacidade do cache, não o tamanho da entrada.

## Velocidade medida — LFM2.5 (experimental)

Medições no mesmo PC Windows com 64 GB de RAM do sistema, o LFM2.5-24B-A2B Q4_K_M oficial, KV FP16, sem MTP ou Vision, e CUDA Graphs ativado. Foram usados os padrões atuais da API: raciocínio ativado, `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`. A capacidade de contexto, os blocos de entrada (256 tokens em todos os casos) e a distribuição dos especialistas foram selecionados automaticamente.

| VRAM disponível | Capacidade de contexto | Geração FP16 | Notas |
| --- | ---: | ---: | --- |
| 24 GB | 128K | 151.71 tok/s | Todos os pesos residentes na VRAM |
| Aproximadamente 16 GB | 128K | 111.71 tok/s | Cálculo de especialistas na GPU e na CPU |
| Aproximadamente 12 GB | 128K | 70.80 tok/s | Cálculo de especialistas na GPU e na CPU |
| Aproximadamente 8 GB | 64K | 51.20 tok/s | Cálculo de especialistas na GPU e na CPU |

Os mesmos três textos de entrada do teste do Qwen foram executados duas vezes cada, gerando 128 tokens por pedido, incluindo o raciocínio. A velocidade é o total de tokens gerados dividido pelo tempo total de decodificação, sem o carregamento e o processamento de entrada. As linhas com memória reduzida limitam a VRAM disponível da mesma GPU de 24 GB e não representam GPUs físicas de 8/12/16 GB. O modelo e as configurações de amostragem diferem dos da tabela do Qwen.

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

Este é um diagrama de distribuição no modo padrão de imagens na GPU, não uma partição de tamanho fixo. A cada carregamento do modelo, o TinyStrata inicia e aquece o codificador de visão da GPU antes de carregar o modelo de linguagem; seus pesos e buffers de trabalho aquecidos permanecem residentes. Em seguida, aloca o cache KV, os demais pesos e os buffers de trabalho, e dimensiona o cache de especialistas conforme a VRAM livre restante. No Windows, usa o menor valor entre a memória livre informada pelo CUDA e a VRAM dedicada livre informada pelo driver NVIDIA, quando ambos estão disponíveis.

O conteúdo do cache muda conforme diferentes especialistas são usados; a capacidade do cache é recalculada quando o modelo é recarregado. Outros aplicativos abertos posteriormente ainda podem consumir VRAM.

A alocação de especialistas normalmente visa uma margem de 768 MiB. Como o Strata, quando falta pouco para todos os especialistas do modelo principal caberem na VRAM, ela pode liberar as cópias empacotadas dos pesos Q8, reduzir um buffer de pontuações de Attention quando existente e diminuir a margem para 256 MiB; os especialistas MTP podem executar na CPU. `--vram-reserve-mib N` fixa a reserva explicitamente. Esse é um objetivo na alocação, não uma garantia permanente de memória livre.

Mais especialistas residentes na GPU geralmente significam menos trabalho para a CPU. No modo híbrido, a RAM do sistema também mantém cópias de apoio dos especialistas armazenados em cache na VRAM, portanto adicionar VRAM não reduz o uso de RAM byte por byte. Deixe RAM também para o sistema operacional e outros aplicativos. Uma maior **capacidade de RAM** permite manter os dados do modelo, mas a velocidade também depende da CPU, da largura de banda da RAM e da GPU. A "memória de GPU compartilhada" do Windows é diferente do cálculo deliberado de especialistas na CPU feito pelo TinyStrata.

O cache KV permanece na VRAM; a RAM do sistema armazena os dados dos especialistas para esse mecanismo de transferência de trabalho. Portanto, uma conversa longa ainda precisa de memória de GPU suficiente para seu cache KV.

## Configurações automáticas — Qwen3.6

Novas configurações escolhem KV/Attention e a capacidade de contexto automaticamente. Você não precisa selecionar uma receita para 8 GB, 12 GB ou 16 GB.

Uma RTX 3060 seleciona INT8. Ambientes compatíveis com RTX das séries 40/50 podem selecionar FP8 nativo; o mecanismo verifica os kernels compilados e os recursos da GPU, além da geração do hardware. Se o FP8 nativo não estiver disponível, o modo automático usa INT8. Isso escolhe um caminho compatível sem executar um teste de velocidade ou qualidade na inicialização. **Attention com FP8 nativo continua experimental.**

O mecanismo nativo estima o tamanho dos blocos de entrada na inicialização, escolhendo 512, 1024, 2048 ou 4096 tokens conforme a VRAM livre, a capacidade de contexto e se os especialistas cabem na GPU. Blocos maiores podem reduzir transferências repetidas de pesos da RAM para a GPU durante o processamento da entrada. `run-api.bat --chunk N` fixa o tamanho; `run-api.bat --chunk auto` restaura a seleção automática nativa.

A capacidade de contexto é o espaço máximo reservado para a entrada e a saída gerada em conjunto. Uma capacidade maior usa mais memória KV mesmo com uma conversa curta, deixando menos VRAM para os especialistas. A capacidade também não implica velocidade igual para todos os comprimentos de conversa: processar um documento longo de fato exige mais trabalho. Como referência, apenas o KV em INT8 ocupa cerca de **0.71 GiB em 64K, 1.42 GiB em 128K e 2.84 GiB em 256K**, incluindo a camada MTP; os demais pesos e buffers de trabalho são adicionais. Nesta implementação, o FP8 nativo usa aproximadamente o mesmo armazenamento KV que o INT8; seu benefício é um caminho de cálculo diferente, em vez de um cache substancialmente menor.

VRAM livre significa a memória disponível no momento do carregamento, não o número impresso na caixa da GPU. Outros aplicativos podem, portanto, fazer com que um contexto menor seja selecionado. Se um recarregamento reduzir a capacidade, a API rejeitará uma entrada que não caiba mais, em vez de truncá-la silenciosamente. No Windows, a seleção automática de contexto consulta tanto a memória livre do CUDA quanto a leitura de todo o dispositivo informada pelo driver da NVIDIA, e usa o menor valor quando ambas as consultas são bem-sucedidas.

A quantização dos pesos GGUF e a precisão do KV são independentes. O modelo UD-IQ3_XXS selecionado tem pesos comprimidos; INT8 ou FP8 descreve o cache da conversa e seu caminho de Attention. O KV de menor precisão pode alterar as respostas, portanto esses modos não prometem uma saída idêntica à do KV em FP32. A implementação mantém todas as posições visíveis do contexto e todos os especialistas selecionados pelo modelo.

Novas configurações do inicializador usam MTP 3, que propõe vários tokens adiante e os verifica com o modelo principal. Isso pode melhorar a velocidade de geração, mas o ganho depende de quantas propostas são aceitas. A verificação busca preservar o comportamento de decodificação do modelo principal; diferenças nos caminhos de execução numérica ainda podem alterar os tokens, portanto ativar MTP não promete uma saída idêntica à obtida ao desativá-lo.

O MTP oferece suporte tanto à geração gulosa quanto à amostragem baseada em temperatura, incluindo penalidades de repetição, frequência e presença. Nas solicitações com amostragem, o modelo principal faz a amostragem de cada linha de verificação, e apenas o prefixo do rascunho que coincide é aceito.

Configurações explícitas substituem as escolhas automáticas, e as configurações existentes selecionadas manualmente continuam em vigor. Por exemplo, `run-api.bat --kv-type int8` fixa INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restaura a seleção automática de KV, contexto e alocação de especialistas.

## Servidor de API

**O TinyStrata funciona apenas por API.** Ele fornece APIs Chat Completions (`/v1/chat/completions`) e Responses (`/v1/responses`) compatíveis com OpenAI e uma API Messages (`/v1/messages`) compatível com Anthropic, incluindo geração em streaming. Conecte um cliente de API ou uma interface externa; o TinyStrata não inclui uma interface de chat.

Para o Qwen, o raciocínio fica ativado por padrão, com `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` e um máximo de **32,768 tokens de saída**, incluindo o raciocínio. As configurações de cada solicitação têm prioridade.

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
