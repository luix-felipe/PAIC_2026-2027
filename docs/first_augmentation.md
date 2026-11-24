# Primeiro passo: anotar e inspecionar as transformações

Temos 23 fotos de **três peças físicas**, uma de cada modelo. A classe visual é
`motherboard`, e a unidade anotada é o painel inteiro. O `piece_id` identifica a
peça física, não uma classe de rede. Top e bottom da mesma peça recebem o mesmo
ID. Ainda precisamos de outras peças e imagens sem motherboard para ampliar a
avaliação.

Esta etapa prepara anotações e demonstra transformações. Não treina modelos,
não cria milhares de cópias e não usa os resultados como medida de desempenho.
O experimento de leitura de Data Matrix continua separado.

## Catálogo atual

As 23 fotos já foram catalogadas por inspeção visual do assistente, com uma
caixa envolvendo a extensão visível do painel inteiro em cada foto:

| Peça física | Descrição visual | side_A | side_B | Fotos |
|---|---|---|---|---|
| `piece_01` | Painel azul | 4 | 0 | 4 |
| `piece_02` | Placa verde | 4 | 4 | 8 |
| `piece_03` | Painel preto | 6 | 5 | 11 |

`side_id` distingue as faces visualmente, sem afirmar qual é top ou bottom.
Por isso, `face` permanece `unknown`. `annotation_source: assistant_visual`
identifica a origem das marcações, e `status: reviewed` registra que as caixas
foram conferidas nas folhas com sobreposição. Não é validação humana
independente nem resultado de um detector treinado.

Sete fotos têm `edge_contact`: o painel encosta em alguma borda e pode estar
parcialmente cortado no enquadramento. Nesses casos a caixa cobre a parte
visível; não inventamos limites de partes fora da imagem. Essas fotos precisam
ser consideradas separadamente ao interpretar os centros e a robustez do modelo.

A primeira marcação feita na interface envolvia somente o Data Matrix. Ela foi
preservada em `datamatrix_roi_candidate`, enquanto `box` passou a representar o
painel. A configuração de leitura `blanks_manual_rois.yaml` continua separada.

Para gerar novamente o CSV e as folhas de conferência:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks catalog
```

Saídas locais: `artifacts/catalog/catalog.csv`, `catalog-overview.jpg` e quatro
folhas `catalog-01.jpg` a `catalog-04.jpg`. O manifesto versionável é
`config/blanks_annotations.yaml`; os relatórios são derivados dele.

## 1. Marcar ou corrigir manualmente

Na raiz do workspace, execute:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks annotate
```

Para cada foto:

1. Arraste uma caixa cobrindo o **painel inteiro**, incluindo sua moldura.
2. Pressione **Enter** para confirmar. **C** cancela a seleção e pula a foto.
3. No terminal, informe `piece_01`, `piece_02` ou `piece_03`. Escolha um ID por
   peça e reutilize-o em todas as fotos dela, inclusive na outra face.
4. Informe a face ou aceite `unknown` se não souber distinguir top/bottom.

Cada marcação é salva imediatamente em `config/blanks_annotations.yaml`.
Ctrl+C encerra o processo e mantém as marcações já salvas. A próxima execução
continua nas fotos pendentes ou em rascunho. Para corrigir uma foto específica:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks annotate --image 20261112_084042.jpg
```

Como o catálogo atual já tem todas as fotos revisadas pelo assistente, use
`--image` para corrigir uma foto ou `--review` para percorrer todas. A interface
registra novas marcações como `annotation_source: manual_gui` e preserva a ROI
candidata de Data Matrix já existente. Os IDs de peça não são nomes comerciais.

## 2. Ver as transformações

Para visualizar as transformações da primeira foto catalogada:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks preview --image 20261112_084042.jpg
```

A saída `artifacts/augmentation-preview.jpg` mostra caixa verde e centro
vermelho; o JSON ao lado registra parâmetros e matrizes para reproduzir cada
transformação. `--allow-draft` serve apenas para visualizar futuras marcações
que ainda estejam em rascunho, sem aprová-las para as divisões.

Comece lendo `scripts/augmentation.py` nesta ordem:

1. `validate_box`: valida os quatro limites da anotação.
2. `letterbox`: reduz a cópia e calcula escala/padding.
3. `transform_points` e `transform_box`: movem os pontos e envolvem os quatro
   cantos numa nova caixa alinhada aos eixos.
4. `augment`: aplica rotação/escala/translação e depois brilho/contraste.

Os limites em `config/augmentation.example.yaml` são **hipóteses ilustrativas**,
não condições medidas da esteira. A visualização usa 320 × 320 por célula; esse
valor não define a entrada do modelo. Não há espelhamento, rotação grande nem
blur nesta primeira política. Uma transformação que corta a caixa do painel é
rejeitada e identificada na comparação.

## 3. Entender as coordenadas

As caixas são `[x0, y0, x1, y1]` em pixels do original orientado pelo EXIF, com
limites superiores exclusivos. A ferramenta converte a seleção da janela para
essa resolução, considerando separadamente os fatores dos dois eixos.

O centro é `((x0+x1)/2, (y0+y1)/2)`. A matriz afim move os quatro cantos e o
centro. Essas caixas são anotações do painel para supervisão; FOMO não passa a
retornar dimensões de caixa por ser treinado com elas.

As transformações atuam na cópia reduzida para estudar a entrada do detector.
Os originais não são sobrescritos nem usados pelo decoder nesta etapa. A ROI
do Data Matrix deve continuar sendo extraída do original no pipeline de leitura.

## 4. Gerar os grupos somente depois de revisar

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks folds
```

O comando exige todas as fotos revisadas e os três grupos físicos. Gera três
rodadas exploratórias, treinando em duas peças e validando na terceira. Todas
as faces e fotos de uma peça ficam juntas. Como há uma peça por modelo físico,
cada rodada deixa também um layout não visto para validação.

A política de augmentation do futuro treinamento será aplicada **somente às
imagens do treino após esta divisão**. Não há conjunto de teste independente
representativo neste acervo, e não devemos ajustar parâmetros repetidamente e
apresentar a validação como um teste final intocado.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes incluem alinhamento entre pixels e anotações, rotação, translação,
escala, padding, recusa de cortes, preservação do original, geração reproduzível
e ausência de vazamento entre grupos. A interação com o mouse é simulada nos
testes; a conferência deste catálogo foi feita pelo assistente nas folhas com
sobreposição. Raspberry Pi, câmera e treinamento permanecem sem validação.

Referências: [transformações geométricas do OpenCV](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
[selectROI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html),
[validação por grupos](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
