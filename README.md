# PAIC 2027 — Edge AI para inspeção de motherboards

O estudo é conduzido por **mentoria guiada pelo chat**, com previsão, tentativa,
verificação e interpretação em passos pequenos. O formato e o ponto de retomada
estão na seção 12 do [AGENTS.md](AGENTS.md). A [trilha didática](docs/study_workflow.md)
e a [preparação das ferramentas](docs/tools_setup.md) servem de apoio; os notebooks
de aulas e laboratórios são materiais locais opcionais, ignorados pelo Git.
Os exemplos e comandos devem ser adaptados à máquina em uso.

Primeiro experimento: avaliar a leitura de Data Matrix nas fotos de
`Datasets/Blanks`, antes de integrar o reconhecimento por IA.

Para iniciar a anotação manual dos painéis e visualizar data augmentation,
consulte o [guia passo a passo](docs/first_augmentation.md). As 23 fotos são de
três peças físicas; o novo manifesto mantém essa identificação separada da
classe visual `motherboard`. Nenhum treinamento foi iniciado.

As 23 caixas de painel já foram catalogadas e conferidas visualmente pelo
assistente: 4 fotos da peça azul, 8 da verde e 11 da preta. Para exportar CSV e
folhas de conferência, execute `python -m scripts.prepare_blanks catalog`.
O catálogo preserva a marcação anterior do Data Matrix como ROI candidata e
sinaliza contato com bordas em sete fotos.

Decisões de escopo confirmadas:

- Uma única classe visual: `motherboard`, incluindo os três modelos de Blanks.
- O painel inteiro constitui uma unidade, mesmo contendo mais de uma placa.
- O Data Matrix equivalente de top ou bottom pode ser usado para a leitura.
- A identificação individual dos três modelos não é necessária nesta etapa.

O ensaio atual usa **ROIs manuais em quatro fotos**. Ele não reconhece uma
motherboard automaticamente e não constitui o MVP. Classe única não fornece,
por si só, a posição do código nos três layouts. A estratégia de localização
automática ainda precisa ser avaliada.

## Executar no PC

Exemplo de preparação em PowerShell, a partir da raiz do workspace:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
.\.venv\Scripts\python.exe scripts/experiment_blanks.py --variants raw clahe otsu
.\.venv\Scripts\python.exe -m pytest -q
```

Os três tratamentos são independentes: imagem sem tratamento, contraste local
CLAHE e limiarização Otsu. O padrão da CLI é somente `raw`. Os parâmetros de
CLAHE são exploratórios; não foram calibrados para produção.

O relatório local `artifacts/blanks-report.json` contém estados, quantidade de
símbolos e tempos por tentativa, sem o conteúdo real dos códigos. O decoder
disponibiliza texto e bytes em memória para uma integração futura.

## Convenção das imagens e das ROIs

`config/blanks_manual_rois.yaml` define `[x0, y0, x1, y1]` em pixels do frame
original **após aplicar a orientação EXIF**, com limites superiores exclusivos.
`frame_size` é `[largura, altura]` e impede o uso da calibração em resolução
diferente. Não há resize no ensaio, nem no decoder. ROIs fora da imagem falham
explicitamente; nunca são truncadas. A captura é lida em cinza, preservando a
resolução espacial, e os tratamentos ocorrem apenas nos recortes.

As ROIs são específicas das quatro fotos, escolhidas por inspeção visual.
Não são offsets por modelo nem resultado de detecção. As demais 19 fotos ainda
não estão calibradas. Uma foto tem 4000 × 3000 pixels; as outras 22 têm
8160 × 6120 antes da orientação EXIF. Essas resoluções não são uma escolha da
câmera futura e exigem memória que deverá ser reavaliada no dispositivo.

## Interpretação científica e próximos passos

- `READ` significa que o ZXing-C++ retornou um Data Matrix válido. Ainda falta
  referência independente para verificar se é o código desejado e se os bytes
  correspondem ao valor esperado. Ausência de leitura permanece explícita.
- A métrica é quantidade de imagens com leitura, não quantidade de produtos
  físicos. As fotos próximas podem retratar a mesma peça; não são amostras
  independentes para estimar generalização.
- Tempos deste ensaio são medições locais no PC, sem metas de FPS e sem
  extrapolação para a Raspberry Pi 3. Não incluem inferência nem rede.
- Próxima etapa de IA: anotar o painel inteiro como `motherboard`, acrescentar
  imagens sem placa e separar treino/validação/teste por peça/sessão de captura.
  FOMO continua candidato conforme `AGENTS.md`; nenhum modelo foi treinado ou
  validado e a estratégia de ROI com uma única classe segue em aberto.
- Câmera, sensor, contagem por passagem, fila e API estão fora deste primeiro
  ensaio. Nenhum destes componentes foi simulado como se fosse integração real.

Interface Python consultada: [documentação oficial do ZXing-C++](https://github.com/zxing-cpp/zxing-cpp/tree/master/wrappers/python).

## Primeiro resultado — 23/11/2026

Execução no PC com Python 3.14.3, OpenCV 4.13.0.92 e ZXing-C++ 3.1.1.
As versões testadas estão em `requirements-experiment.txt`; o relatório contém
os tempos e as versões efetivas. O ambiente local inicial reaproveitou os
pacotes já instalados via `venv --system-site-packages`; a instalação em um
ambiente completamente limpo ainda não foi repetida.

| Foto | Sem tratamento | CLAHE | Otsu |
|---|---|---|---|
| `20261112_084042.jpg` | Sem leitura | Leitura | Sem leitura |
| `20261112_084324.jpg` | Sem leitura | Sem leitura | Sem leitura |
| `20261112_084419.jpg` | Leitura | Sem leitura | Leitura |
| `20261112_084440.jpg` | Sem leitura | Sem leitura | Sem leitura |

Considerando qualquer tentativa, 2 das 4 fotos tiveram ao menos uma leitura.
Cada tratamento isolado leu 1 das 4 fotos: CLAHE recuperou uma leitura e perdeu
outra. Não há evidência para torná-lo obrigatório. Não extrapolar estes números
para o conjunto inteiro nem para a esteira, e não interpretá-los como taxa de
leitura correta sem referência do conteúdo esperado.

Os testes usam símbolos sintéticos e cobrem resolução/limites da ROI, ausência
de resize, leitura ECC200, recusa de QR, falha de captura, resolução divergente,
integração com arquivo e relatório sem conteúdo real. Hardware, IA e desempenho
na Raspberry Pi 3 permanecem sem validação.
