# Primeiro passo: anotar e inspecionar as transformações

Temos 23 fotos de **três peças físicas**, uma de cada modelo. A classe visual é
`motherboard`, e a unidade anotada é o painel inteiro. O `piece_id` identifica a
peça física, não uma classe de rede. Top e bottom da mesma peça recebem o mesmo
ID. Ainda precisamos de outras peças e imagens sem motherboard para ampliar a
avaliação.

Esta etapa prepara anotações e demonstra transformações. Não treina modelos,
não cria milhares de cópias e não usa os resultados como medida de desempenho.
O experimento de leitura de Data Matrix continua separado.

## 1. Marcar manualmente

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

O rascunho inicial dessa foto foi marcado visualmente pelo assistente apenas
para demonstrar o processo. `piece_01` é seu identificador provisório, sem nome
comercial de modelo. As outras 22 fotos começam sem caixa nem grupo atribuído.
O rascunho precisa de revisão manual antes de participar das divisões.

## 2. Ver as transformações

Antes de revisar a primeira foto, podemos visualizar explicitamente o rascunho:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_blanks preview --image 20261112_084042.jpg --allow-draft
```

Depois da revisão, o mesmo comando funciona sem `--allow-draft`. A saída
`artifacts/augmentation-preview.jpg` mostra caixa verde e centro vermelho; o
JSON ao lado registra parâmetros e matrizes para reproduzir cada transformação.

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
testes; a revisão das caixas exige uso humano da janela. Raspberry Pi, câmera
e treinamento permanecem sem validação.

Referências: [transformações geométricas do OpenCV](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
[selectROI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html),
[validação por grupos](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
