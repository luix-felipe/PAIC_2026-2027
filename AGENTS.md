# AGENTS.md — Inspeção visual e leitura de Data Matrix em esteira

## 1. Objetivo do projeto

Desenvolver um sistema de visão computacional embarcado, inicialmente em uma **Raspberry Pi 3**, para identificar o modelo de uma placa/produto que passa por uma esteira, localizar a região conhecida de seu **Data Matrix**, decodificar o conteúdo, contabilizar a passagem sem duplicidade e enviar um evento para uma API.

O projeto combina **um modelo leve de visão computacional** com **processamento determinístico de imagem**. Não usar uma rede neural para tarefas que possam ser resolvidas com geometria simples e uma biblioteca de decodificação.

### Premissas confirmadas

- Os produtos estão **em movimento sobre uma esteira**.
- Cada modelo de produto mantém **orientação previsível e fixa** ao passar pela câmera.
- A posição do Data Matrix é **conhecida e específica por modelo**.
- A primeira plataforma embarcada será uma **Raspberry Pi 3**, sem presumir acelerador de IA externo.
- O objetivo é identificar a placa, contar a passagem, ler o Data Matrix e transmitir o resultado para uma API.

### Escopo do experimento atual

- O projeto é uma iniciação científica para 2027. A implementação deve ser **o mais “na mão” possível**: explicar e escrever etapas pequenas, incluindo arquitetura da rede e laço de treinamento, usando frameworks para operações numéricas e diferenciação automática.
- O primeiro detector terá **uma única classe visual, `motherboard`**, abrangendo os três modelos físicos de `Datasets/Blanks`. A unidade anotada é o **painel inteiro**, mesmo contendo várias placas.
- Reconhecer o modelo físico específico não é requisito deste experimento. A arquitetura por modelo descrita adiante permanece como referência de evolução; não supor que uma classe única permite selecionar automaticamente uma ROI diferente para cada modelo.
- Segundo o usuário, o Data Matrix equivalente é gravado em top e bottom; qualquer uma dessas gravações pode ser utilizada. Isso não garante que todo símbolo visível seja o código desejado: a conferência do conteúdo esperado continua pendente.
- **O próximo incremento é o carregador de dados em PyTorch, a geração dos alvos por grade e sua visualização**, conforme a seção 11. A implementação da CNN e os treinamentos comparativos pertencem ao incremento seguinte.

### Variáveis ainda não definidas

Velocidade da esteira, intervalo mínimo entre produtos, modelo exato da Raspberry Pi 3, tipo/resolução de câmera, iluminação, quantidade de modelos no escopo final de produção, dimensão física dos Data Matrix, interface do sensor de presença e contrato da API. O acervo inicial de Blanks contém três modelos físicos.

**Não inventar valores para essas variáveis.** Torná-las configuráveis; usar valores de exemplo apenas se forem identificados explicitamente como ilustrativos.

## 2. Arquitetura de referência

```text
Câmera / disparo por sensor (quando disponível)
                   |
                   v
          Captura do frame original
                   |
          +--------+---------+
          |                  |
          v                  |
    Pré-processamento         |
   (resize/normalização)      |
          |                  |
          v                  |
 Inferência TinyML/Edge       |
 (classe + centro da placa)   |
          |                  |
          v                  |
 Consulta à configuração      |
   de ROI por modelo          |
          |                  |
          +--------+---------+
                   v
        ROI no frame ORIGINAL
                   |
                   v
          Decodificação ECC200
             (ZXing-C++)
                   |
                   v
     Validação e evento único
                   |
                   v
        Fila local de entrega
                   |
                   v
           API HTTP/JSON
```

### Regra central de qualidade de imagem

O redimensionamento serve **somente para a entrada do modelo**. O recorte para decodificação deve ser extraído do **frame original**, preservando a resolução dos módulos do Data Matrix e a *quiet zone*. Nunca decodificar automaticamente a partir da imagem reduzida de inferência.

### Segmentação versus ROI

O termo “segmentação” usado no escopo do projeto refere-se inicialmente a **recorte de região de interesse (ROI)**, e **não a segmentação semântica/pixel a pixel**. Não introduzir máscaras ou uma segunda rede neural sem evidência de que a ROI geométrica é insuficiente.

## 3. Stack tecnológica preferencial

| Função | Escolha inicial | Observação |
|---|---|---|
| Plataforma | Raspberry Pi 3 | CPU, memória e taxa real devem ser medidas |
| Linguagem da aplicação | Python | Prototipagem e integração; otimizar apenas com medição |
| Treinamento de modelo | PyTorch com código próprio | Executar em PC/servidor; SSCMA como referência e comparação |
| Modelo didático inicial | CNN pequena com saída por grade, inspirada em FOMO | Implementar após validar dados e alvos; não presumir equivalência ao FOMO do SSCMA |
| Modelo de referência | FOMO com MobileNetV2 0.35 | Classe e centro aproximado; não retorna bounding box completa |
| Alternativa de modelo | RTMDet compacto ou classificador leve | Avaliar quando FOMO não for suficiente |
| Runtime embarcado | TensorFlow Lite / LiteRT | Confirmar compatibilidade da exportação do modelo escolhido |
| Imagens e ROI | OpenCV + NumPy | Operações determinísticas |
| Data Matrix | ZXing-C++ | Preferência; `libdmtx` é alternativa técnica |
| API | HTTP/JSON | Cliente isolado do pipeline visual |
| Configuração | YAML | Modelos, ROI, parâmetros e thresholds |
| Testes | pytest | Mockar câmera, modelo, decodificador e rede |

Não presumir que um modelo exportado pelo SSCMA rode na Raspberry Pi 3 sem conversão/validação. Registrar dependências, versões testadas, formato de entrada/saída, quantização e operações suportadas pelo runtime. O treinamento é feito fora do dispositivo; a Pi realiza **apenas inferência**.

A adoção de um fluxo próprio em PyTorch foi solicitada pelo usuário para compreender e controlar arquitetura, dados, perda, otimização e avaliação. SSCMA permanece como referência técnica. Isso não altera o alvo de inferência TFLite/LiteRT, cuja conversão e compatibilidade ainda precisam ser demonstradas. Manter dependências de treinamento separadas das dependências do leitor e da aplicação embarcada; não instalar ferramentas de treinamento na Pi por conveniência.

## 4. Contratos entre módulos

Cada componente deve possuir uma interface clara e independente do hardware.

### 4.1 Aquisição

Entrada: câmera ou fonte de vídeo; opcionalmente, trigger de sensor fotoelétrico.

Saída: `frame_original`, `captured_at`, `capture_id` e metadados da câmera.

- Não bloquear indefinidamente quando uma captura falhar.
- Preservar o frame original até o término da extração de ROI.
- A primeira implementação deve aceitar também imagens e vídeos gravados, permitindo desenvolvimento sem a câmera física.

### 4.2 Inferência

Entrada: cópia pré-processada do frame.

Saída mínima: `model_id`, `confidence`, `center_x`, `center_y` e informações da transformação de coordenadas.

- FOMO retorna **centro aproximado**, não dimensões reais de caixa delimitadora.
- Converter as coordenadas corretamente para o frame original, respeitando escala, crop, padding e *letterboxing*.
- Se a confiança for insuficiente ou o modelo for desconhecido, registrar estado explícito; não inferir uma ROI arbitrária.
- Caso um classificador substitua FOMO em uma estação com placa centralizada, documentar a hipótese de posicionamento fixo.
- No experimento de classe única, distinguir `class_id` (`motherboard`) de `model_id` específico e de `piece_id` do dataset. **Não usar `piece_id` conhecido durante a anotação como se fosse uma identificação produzida na inferência.**

### 4.3 Localização da ROI

Entrada: `frame_original`, `model_id`, centro detectado e configuração de ROI.

Saída: imagem de ROI e suas coordenadas no frame original.

- Cada modelo possui uma ROI calibrada, expressa em deslocamentos a partir do centro **ou** coordenadas normalizadas; escolher uma convenção e documentá-la.
- Validar os limites do frame antes de fazer o recorte. **Não truncar silenciosamente** códigos cortados.
- Preservar margem ao redor do Data Matrix, inclusive *quiet zone*.
- Separar configuração geométrica de lógica de decodificação.
- Se houver variação relevante de escala, não usar offsets fixos em pixels sem calibração; considerar ROI relativa à caixa do objeto ou calibração da câmera.

Exemplo **ilustrativo** de configuração YAML:

```yaml
models:
  placa_A:
    roi:
      reference: detection_center
      dx_px: 90
      dy_px: -30
      width_px: 110
      height_px: 110
  placa_B:
    roi:
      reference: detection_center
      dx_px: -75
      dy_px: 45
      width_px: 100
      height_px: 100
```

### 4.4 Decodificação

Entrada: ROI em resolução original.

Saída: `decoded_text`/`decoded_bytes`, status de leitura e, quando houver, metadados do símbolo.

- Priorizar **Data Matrix ECC200**.
- Encapsular ZXing-C++ atrás de uma interface que permita trocar por libdmtx sem alterar o restante da aplicação.
- Não inventar conteúdo quando a leitura falhar.
- Possíveis melhorias (contraste, binarização, nova captura) devem ser opcionais, mensuradas e configuráveis.
- Validar o conteúdo conforme as regras reais do produto somente quando essas regras forem disponibilizadas.

### 4.5 Contagem e ciclo de vida do produto

O sistema deve criar **um único evento por passagem física**. Não contar o mesmo produto novamente por aparecer em vários frames.

Estados sugeridos:

```text
DETECTADO -> CAPTURADO -> CLASSIFICADO -> DECODIFICADO -> REGISTRADO
                                      \-> FALHA_LEITURA -> REGISTRADO
                    \-> FALHA_CLASSIFICACAO -> REGISTRADO
```

- Contagem física e contagem de leituras bem-sucedidas são métricas distintas.
- Quando disponível, preferir o sensor fotoelétrico como fonte de eventos de passagem, com debounce e associação entre trigger e captura.
- Sem sensor, implementar lógica de rastreamento/linha virtual que seja testável; nunca incrementar contador a cada inferência por frame.
- Não presumir que o conteúdo do Data Matrix é único entre produtos.
- A lógica deve tolerar falhas de captura, classificação e leitura sem perder o registro da passagem detectada.

### 4.6 API e persistência

- Enviar registros com HTTP/JSON através de um módulo isolado, com timeout, retentativas limitadas e backoff.
- A captura e a inferência **não podem depender da disponibilidade da API**.
- Usar fila local persistente (por exemplo, SQLite) para entregas pendentes quando necessário.
- Gerar `event_id` único por passagem e permitir idempotência no servidor.
- Não incluir chaves, tokens ou URLs privadas hardcoded no repositório.
- O esquema exato da API é **pendente de definição**; usar o payload a seguir apenas como contrato provisório:

```json
{
  "event_id": "station-01-000184",
  "device_id": "raspberry-pi-3",
  "model_id": "placa_A",
  "datamatrix": "ABC123456789",
  "status": "LEITURA_OK",
  "captured_at": "2026-11-23T13:00:00-04:00"
}
```

Em caso de falha de leitura, `datamatrix` deve ser `null`, não uma string inventada.

## 5. Estrutura sugerida do repositório

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── config/
│   ├── example.yaml
│   └── roi_models.example.yaml
├── models/
│   └── README.md              # origem, versão, licença e formato dos modelos
├── src/
│   └── inspection/
│       ├── capture.py         # câmera, vídeo e imagens
│       ├── preprocessing.py   # resize, normalização, coordenadas
│       ├── detector.py        # interface TFLite e saídas do modelo
│       ├── roi.py             # calibração e recorte
│       ├── decoder.py         # ZXing-C++ / libdmtx
│       ├── tracking.py        # eventos e deduplicação
│       ├── api_client.py      # entrega HTTP e retentativas
│       ├── queue_store.py     # fila local persistente
│       ├── pipeline.py        # orquestração
│       └── main.py            # CLI / inicialização
├── scripts/
│   ├── calibrate_roi.py
│   └── benchmark.py
├── tests/
│   ├── test_coordinates.py
│   ├── test_roi.py
│   ├── test_decoder.py
│   ├── test_tracking.py
│   ├── test_api_client.py
│   └── test_pipeline.py
└── docs/
    ├── architecture.md
    ├── hardware.md
    └── benchmarks.md
```

A estrutura é uma **proposta inicial**, não uma obrigação de criar arquivos vazios. Adaptar à estrutura real do repositório, preservando responsabilidades claras.

## 6. Instruções obrigatórias para agentes de código

1. **Inspecione o repositório antes de alterar arquivos.** Leia o README, dependências, configuração, testes e decisões já implementadas. Não substitua módulos existentes apenas para seguir a árvore sugerida.
2. **Trabalhe em incrementos pequenos e verificáveis.** Primeiro pipeline com imagem gravada; depois modelo; depois câmera e esteira; depois API.
3. **Preserve a arquitetura híbrida.** Não treine uma rede para ler o conteúdo do Data Matrix; use decodificador próprio. Não acrescente segmentação por máscara sem justificativa técnica.
4. **Priorize desempenho na Raspberry Pi 3.** Evite dependências de treinamento pesadas no dispositivo, cópias desnecessárias de frames e processamento em resolução integral fora da ROI.
5. **Não acople hardware ao domínio.** Interfaces de câmera, GPIO, modelo de IA e API devem ter substitutos simulados para testes em PC.
6. **Mantenha coordenadas explícitas.** Documente unidades, resolução de referência, transformações e convenção do centro da ROI.
7. **Trate falhas como resultados esperados.** Sem produto, modelo desconhecido, baixa confiança, ROI inválida, Data Matrix ilegível, timeout de API e câmera desconectada devem ter estados claros.
8. **Não invente integração de hardware.** Se não houver acesso à câmera, esteira, sensor ou modelo real, implemente mocks e explique precisamente o que ainda exige validação física.
9. **Adicione testes para toda regra crítica.** Especialmente conversões de coordenadas, limites da ROI, ausência de duplicidade na contagem e retentativa de envio.
10. **Não mude a stack sem motivação.** Preferência atual: treinamento próprio em PyTorch com SSCMA como referência, inferência TFLite na Pi, OpenCV para ROI, ZXing-C++ para leitura e HTTP/JSON para API. A motivação do fluxo próprio é a implementação didática solicitada pelo usuário. Outras mudanças devem registrar motivo e impacto.
11. **Não inclua segredos nem dados de produção reais** em código, logs, testes ou exemplos. Use `.env.example` ou variáveis de ambiente quando necessário.
12. **Ao concluir uma alteração, informe** o que mudou, como executar, quais testes passaram, o que não foi testado e eventuais limitações de hardware.

## 7. Testes e critérios de aceitação

### Testes unitários mínimos

- Conversão de coordenadas de inferência para resolução original, incluindo padding e redimensionamento não proporcional.
- ROI correta para diferentes modelos e centros detectados.
- Falha explícita em ROI fora do frame ou modelo desconhecido.
- Decodificação de imagens de teste com Data Matrix válido e resultado de falha para imagem sem código.
- Um único evento/contagem para múltiplos frames do mesmo produto.
- Distinção entre produto contado, código lido e evento enviado.
- Retentativa de envio sem gerar novo `event_id`.

### Integração mínima

- Rodar um pipeline de ponta a ponta com imagens gravadas e detector simulado (antes de usar modelo real).
- Registrar logs estruturados por `event_id`/`capture_id`, evitando dados sensíveis desnecessários.
- Demonstrar que uma API indisponível não bloqueia o processamento visual.
- Testar a mesma lógica tanto no PC quanto na Raspberry Pi 3, quando o hardware estiver disponível.

### Benchmark obrigatório na Raspberry Pi 3

Medir, por produto:

- Tempo de captura, pré-processamento, inferência, extração de ROI e decodificação.
- Latência total e percentis **p50/p95**, além da vazão em produtos/minuto.
- Uso máximo de RAM e consumo aproximado de CPU.
- Taxa de detecção/classificação correta e taxa de **leitura correta de Data Matrix**.
- Eventos perdidos, leituras falhas e registros duplicados.
- Quantização e resolução de entrada efetivamente testadas.

Não estabelecer metas de FPS, precisão ou consumo como fatos sem dados reais da esteira. Definir metas depois de medir velocidade, espaçamento e requisitos de produção.

## 8. Plano de implementação

### Fase 1 — Prova de conceito sem IA

1. Capturar ou carregar imagens reais das placas.
2. Definir ROI manual por modelo.
3. Decodificar os Data Matrix com ZXing-C++.
4. Medir qualidade e taxa de leitura sob iluminação e movimento reais.

**Saída esperada:** leitor de Data Matrix confiável em imagens reais, independente da IA.

### Fase 2 — Identificação automatizada

1. Preparar o dataset de classe única `motherboard`, mantendo peça física e face como metadados e separando os grupos antes da augmentation.
2. Implementar e conferir o carregador de dados, as transformações e os alvos por grade, conforme a seção 11.
3. Implementar a CNN didática e o laço de treinamento em PyTorch; começar com uma execução de verificação e depois comparar sem/com augmentation sob condições documentadas.
4. Reproduzir FOMO/MobileNetV2 0.35 como referência, registrando diferenças de arquitetura, inicialização e treinamento. Não atribuir a um modelo próprio resultados de outro projeto.
5. Exportar e validar o modelo com runtime TFLite compatível e converter corretamente as coordenadas para a imagem original.
6. Avaliar a localização geométrica das ROIs nos três layouts. A estratégia automática com uma única classe ainda está em aberto; não inferir uma ROI arbitrária a partir do centro.

**Saída esperada:** identificação + extração automática da ROI + leitura.

### Fase 3 — Integração embarcada na esteira

1. Integrar câmera e, se disponível, sensor de presença/trigger.
2. Implementar máquina de estados de passagem e deduplicação.
3. Registrar eventos localmente, inclusive falhas de leitura.
4. Integrar fila e envio para API.
5. Medir tempos e ajustar resolução, modelo e taxa de captura.

**Saída esperada:** ciclo de inspeção completo na Raspberry Pi 3.

### Fase 4 — Pesquisa e otimização

Comparar o sistema com baselines de classificação simples e, se necessário, detecção com bounding boxes. Analisar trade-offs entre acurácia, taxa de leitura, latência, memória e consumo. Só introduzir métodos mais complexos após identificar limitações mensuráveis da solução inicial.

## 9. Ordem de prioridade técnica

1. **Legibilidade do Data Matrix na imagem original** (iluminação, foco, exposição, resolução).
2. **Sincronização da captura e passagem física.**
3. **ROI correta para cada modelo.**
4. **Leitura e contagem sem perda ou duplicidade.**
5. **Desempenho da inferência TinyML na Raspberry Pi 3.**
6. **Entrega robusta à API.**

Uma maior acurácia do detector não compensa um Data Matrix desfocado ou uma lógica que conta o mesmo produto várias vezes.

## 10. Definição de pronto para o primeiro MVP

Considerar o MVP concluído somente quando for possível:

- Processar uma imagem ou captura real de placa.
- Reconhecer `motherboard` com o detector integrado ou sinalizar falha explicitamente; identificar o modelo físico específico quando essa tarefa fizer parte do escopo adotado.
- Selecionar automaticamente a ROI correta na resolução original.
- Decodificar o Data Matrix ou informar falha sem dados fictícios.
- Gerar exatamente um registro por passagem detectada.
- Persistir e transmitir eventos para API, inclusive após indisponibilidade temporária da rede.
- Reproduzir testes automatizados e apresentar resultados de benchmark na Raspberry Pi 3.

**Princípio orientador:** começar simples, medir no hardware real e acrescentar complexidade somente quando houver uma limitação demonstrada.

## 11. Próximo incremento — Dataset, alvos por grade e visualização

### 11.1 Estado existente a preservar

- `Datasets/Blanks` contém **23 fotos de três peças físicas**, uma de cada modelo: `piece_01` (azul, 4 fotos), `piece_02` (verde, 8 fotos) e `piece_03` (preta, 11 fotos). Fotos e versões aumentadas não constituem novas peças independentes.
- A fonte das anotações é `config/blanks_annotations.yaml`. `box` representa a extensão visível do **painel inteiro**, em `[x0, y0, x1, y1]`, pixels do original orientado pelo EXIF e limites superiores exclusivos.
- As 23 caixas foram conferidas visualmente pelo assistente, com `status: reviewed` e `annotation_source: assistant_visual`. Esse estado não representa uma revisão humana independente ou validação de um detector.
- `piece_id` serve para agrupamento físico, não como classe de treinamento. `side_id` (`side_A`/`side_B`) distingue faces visualmente; `face` permanece `unknown` quando top/bottom não foi confirmado. Todas as faces da mesma peça pertencem ao mesmo grupo.
- Sete fotos têm `edge_contact`, indicando possível corte no enquadramento. Preservar a indicação e explicitar sua inclusão ou exclusão no experimento; não inventar a extensão da placa fora do frame.
- A marcação anterior do usuário na primeira foto foi preservada em `datamatrix_roi_candidate`. Não confundir essa ROI com a caixa de supervisão do painel, nem assumir que o conteúdo do código já foi validado.
- Reutilizar `scripts/augmentation.py` para geometria e `scripts/prepare_blanks.py` para anotação, catálogo e divisões. Os relatórios em `artifacts/` são derivados e ignorados pelo Git; uma instalação nova deve conseguir regenerá-los a partir do manifesto, sem depender de arquivos locais ausentes.
- `scripts/experiment_blanks.py` e `config/blanks_manual_rois.yaml` são o ensaio separado de leitura em quatro fotos, com leitura obtida em duas delas. Não há detector treinado nem benchmark na Pi 3.

### 11.2 Entrega deste incremento

Implementar um carregador de dados em PyTorch e uma visualização de amostras, sem antecipar a implementação completa da rede:

1. Ler imagens e anotações do manifesto, validando arquivo, resolução orientada, caixa, estado de revisão e peça física. Carregar imagens sob demanda; evitar manter todos os frames de alta resolução na RAM.
2. Consumir ou regenerar três divisões por peça: treinar em duas e validar na terceira. Preservar todas as fotos e faces de uma peça no mesmo grupo. Nenhuma versão aumentada pode atravessar a divisão.
3. Aplicar resize com padding em uma cópia destinada ao detector. Tornar resolução de entrada, fator de redução da grade e parâmetros configuráveis; os 320 × 320 usados na visualização existente **não são uma resolução de treinamento já escolhida**.
4. Aplicar augmentation somente no treino, com sementes controladas e parâmetros registrados. A validação recebe apenas o pré-processamento determinístico necessário à entrada. Comparações sem/com augmentation devem usar as mesmas divisões.
5. Transformar caixa e centro com a mesma geometria da imagem e registrar as transformações original → entrada → imagem aumentada. Preservar a possibilidade de mapear coordenadas para o original. Gerar os alvos **depois** dessas transformações.
6. Para o detector didático, gerar uma grade inteira: `0` para fundo e `1` na célula do centro de `motherboard`. Definir e documentar indexação `[linha_y, coluna_x]`, formato do tensor e regra de borda. Rejeitar dimensões incompatíveis com o fator da grade e centros inválidos; não corrigir erros de coordenadas por truncamento silencioso.
7. Disponibilizar, por amostra, imagem de entrada, grade-alvo, caixa e centro transformados, `piece_id`, caminho de origem, flags de borda e metadados de transformação. Documentar canais, ordem dos eixos, dtype e normalização; não tratar imagens BGR do OpenCV como RGB por engano.
8. Gerar uma visualização reproduzível com imagem, caixa, centro e grade esperada, incluindo exemplos de treino aumentado e validação determinística. Ela demonstra a preparação dos dados, não uma predição do modelo.

Essa grade é **supervisão por centro para o modelo didático**, não uma máscara semântica do painel. A anotação com caixas não faz FOMO retornar dimensões de caixa. Documentar diferenças em relação aos alvos e à perda da implementação de referência no SSCMA.

### 11.3 Critérios de aceitação

- Um comando documentado carrega os dados do manifesto e gera as visualizações no PC, sem exigir câmera, GPIO, API ou um modelo treinado.
- Testes verificam ausência de vazamento por peça, aplicação de augmentation somente ao treino, alinhamento entre imagem/caixa/centro/grade, padding e arredondamento, células nas bordas, falhas de entrada e reprodução com a mesma semente.
- A visualização é conferida antes de usar os alvos em treinamento. Manter o leitor ZXing-C++ separado e a extração do Data Matrix na resolução original.
- Não apresentar métricas de generalização a partir das 23 fotos como evidência sobre novas peças de produção. Com uma peça por modelo, a validação por peça também deixa um layout inteiro fora do treino. Não existe, neste acervo, um teste final independente representativo.
- Registrar o que foi executado, versões de treinamento realmente testadas e limitações. Não presumir que a versão de Python do leitor seja compatível com todas as dependências de treinamento.

### 11.4 Trabalho seguinte e frentes paralelas

Após conferir o carregador e os alvos, implementar a CNN pequena, a perda e o laço de treinamento, iniciando por uma execução de verificação. Depois realizar a comparação sem/com augmentation e com FOMO/MobileNetV2 0.35. Fixar e registrar orçamento, sementes e critérios de avaliação; não copiar hiperparâmetros de artigos como se estivessem validados para este acervo.

Em paralelo, coletar imagens sem motherboard para medir falsos positivos, obter outros exemplares físicos e investigar as falhas de Data Matrix nas imagens originais. O avanço do detector não substitui a validação da leitura, da contagem por passagem ou do desempenho no hardware.

## 12. Mentoria guiada pelo chat

### 12.1 Formato de trabalho confirmado

- O usuário escolheu **mentoria guiada pelo chat**, em passos pequenos, para compreender e implementar o projeto. Conduzir o estudo pela conversa; criar novas aulas, laboratórios ou geradores de notebooks somente quando solicitado.
- Usar o repositório didático `luix-felipe/VC_iREDE` como referência de método: previsão, exemplo numérico, tentativa própria, verificação e interpretação. Adaptar o conteúdo ao escopo do PAIC; escolhas daquele repositório sobre defeitos, YOLO ou máscaras não alteram a arquitetura deste projeto.
- A cada bloco, explicar o objetivo, a entrada, a saída e as unidades; apresentar uma operação pequena; pedir uma previsão ou tentativa; conferir o resultado com o usuário e então propor o passo seguinte. Não entregar uma sequência extensa de soluções quando o usuário estiver praticando uma etapa.
- Explicar onde encontrar a ferramenta na documentação oficial e como executar a operação necessária. Relacionar cada parâmetro ao efeito esperado, incluindo um caso inválido quando isso ajudar a entender a regra.
- Distinguir código existente, demonstração executada pelo assistente, exercício proposto e resultado efetivamente executado/interpretado pelo usuário. Uma pergunta ainda sem resposta não significa compreensão ou atividade concluída.
- Os materiais locais em `notebooks/aulas/` e `notebooks/laboratorios/` ficam ignorados pelo Git. Podem ser consultados se disponíveis, mas a mentoria e a reprodução do projeto devem funcionar sem eles e sem os arquivos derivados de `artifacts/`.

### 12.2 Independência da máquina

- Apresentar exemplos com caminhos relativos à raiz do repositório, sem depender do nome de usuário, drive, pasta absoluta, IDE ou ambiente virtual desta sessão.
- Antes de instruções de instalação ou execução, conferir o Python, o executável e as bibliotecas disponíveis na máquina em uso. Adaptar comandos ao sistema operacional; não presumir que versões ou ambientes de uma sessão anterior estejam presentes.
- Preferir operações que possam ser executadas em um interpretador Python ou em um script pequeno, deixando o uso de notebook como opção. Explicar separadamente comandos de terminal e código Python. Se houver notebook, conferir também seu kernel.
- Manter dependências e versões documentadas por finalidade: estudo no PC, leitor, futuro treinamento e inferência embarcada. Registrar o ambiente realmente testado sem tratá-lo como requisito universal ou compatibilidade já comprovada na Pi 3.
- Manter decisões de escopo, sequência de estudo e ponto de retomada neste arquivo, de modo que outro agente ou uma nova máquina possam continuar sem depender de abas abertas ou arquivos locais ignorados.

### 12.3 Sequência de estudo

1. Representação de imagens e coordenadas: canais, eixos, dtype, ROI, caixa, centro e padding, retomando fundamentos quando necessário.
2. Data augmentation e organização do dataset: alterações fotométricas e geométricas, divisão por peça, sementes e inspeção visual.
3. Carregador PyTorch, transformação das anotações, alvos por grade e visualização, cumprindo a seção 11 antes de treinar.
4. CNN didática, perda, autograd, otimização e laço próprio, começando por uma execução de verificação.
5. Comparações sem/com augmentation e, depois, com FOMO/MobileNetV2 0.35, sob protocolo documentado.
6. Localização validada da ROI e leitura no frame original, mantendo a investigação das falhas de Data Matrix em paralelo.
7. Exportação, inferência e medições na Raspberry Pi 3; integração de passagem, persistência e API conforme as fases do projeto.

Essa sequência organiza a mentoria; não substitui os critérios de aceitação nem implica que as etapas anteriores já tenham sido dominadas pelo usuário.

### 12.4 Ponto de retomada da mentoria

O usuário pediu para avançar para **data augmentation**. A introdução pelo chat apresentou:

- Transformações fotométricas: brilho e contraste alteram intensidades; caixa e centro mantêm suas coordenadas.
- Transformações geométricas: translação, rotação e escala exigem transformar imagem, caixa e centro juntos.
- Augmentation somente no treino, depois da divisão por peça. Versões aumentadas continuam pertencendo à mesma peça física; validação recebe preparação determinística.
- Primeiro exercício de brilho: array `np.array([0, 100, 250], dtype=np.uint8)` com deslocamento ilustrativo `15`. A operação é converter para `float32`, somar, aplicar `np.clip(..., 0, 255)` e converter de volta para `uint8`, conforme o núcleo de `scripts/augmentation.py`.

**Ainda pendente:** o usuário prever os três valores de saída e explicar se o centro anotado muda com o brilho. O exercício foi proposto; sua execução e interpretação pelo usuário não foram confirmadas.

Retomar por essa previsão ou pela dúvida que o usuário trouxer. Depois, comparar deslocamentos positivo e negativo e o efeito da saturação; avançar para contraste e, em seguida, translação, rotação e escala. Usar parâmetros ilustrativos até haver medições da captura real, preservar os originais e conferir o alinhamento das anotações antes de gerar os alvos por grade.
