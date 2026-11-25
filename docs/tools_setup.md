# Ferramentas para estudar e executar o PAIC

Comece com Python, NumPy, OpenCV e um notebook no VS Code. Eles permitem observar
imagens e conferir a geometria antes de instalar o framework de treinamento.
O ambiente de estudo, o leitor já existente e o futuro treino têm dependências
separadas para que uma instalação não altere silenciosamente a outra.

## Onde buscar e o que aprender primeiro

| Ferramenta | Onde obter ou consultar | Uso neste projeto | Primeira operação |
|---|---|---|---|
| Python e `venv` | [Python](https://www.python.org/downloads/), [ambientes virtuais](https://docs.python.org/3/library/venv.html) | Interpretador e isolamento das bibliotecas | Conferir `sys.executable` e criar um ambiente |
| VS Code e Jupyter | [Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks), [seleção de kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) | Editar e executar células | Escolher o Python correto em `Select Kernel` |
| NumPy | [Introdução oficial](https://numpy.org/doc/stable/user/absolute_beginners.html) | Matrizes, índices, tipos e coordenadas | Criar um array e recortar `[y0:y1, x0:x1]` |
| OpenCV | [Imagens](https://docs.opencv.org/4.x/dc/d2e/tutorial_py_image_display.html), [geometria](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) | Leitura, cores, resize e matrizes afins | Abrir uma foto e converter BGR para RGB |
| Matplotlib | [Tutorial oficial](https://matplotlib.org/stable/tutorials/pyplot.html) | Ver imagem, caixa, centro e grade | `imshow` com eixos e unidades explícitos |
| PyYAML | [Documentação oficial](https://pyyaml.org/wiki/PyYAMLDocumentation) | Ler manifestos e parâmetros existentes | `yaml.safe_load` em um arquivo UTF-8 |
| pytest | [Primeiros passos](https://docs.pytest.org/en/stable/getting-started.html) | Testar regras do pipeline | Executar um teste e entender uma falha |
| ZXing-C++ | [Bindings Python oficiais](https://github.com/zxing-cpp/zxing-cpp/tree/master/wrappers/python) | Decodificar Data Matrix na ROI original | Usar o ensaio existente com `--variants raw` |
| PyTorch | [Instalação](https://pytorch.org/get-started/locally/), [tutorial de dados](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) | Dataset, tensores, rede, derivadas e otimização no PC | Criar um tensor e depois implementar `Dataset` |
| SSCMA | [Repositório oficial](https://github.com/Seeed-Studio/ModelAssistant) | Referência posterior para FOMO | Identificar revisão, arquitetura, alvos e perda antes da comparação |
| LiteRT/TFLite | [Documentação oficial](https://ai.google.dev/edge/litert) | Futuro runtime de inferência | Comparar a saída exportada com a saída PyTorch |

As extensões do VS Code são **Python**, da Microsoft (`ms-python.python`), e
**Jupyter**, da Microsoft (`ms-toolsai.jupyter`). Instale-as pela aba Extensions.
O kernel é o processo Python que executa o notebook; ele pode ser diferente do
`python` encontrado pelo terminal. A escolha do kernel deve apontar para o ambiente
em que você instalou as bibliotecas, conforme a [documentação de kernels](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).

Ao consultar uma ferramenta, procure a operação necessária para a tarefa atual.
Leia seu contrato, execute um exemplo pequeno e mude uma entrada. Depois a aplique
à fotografia. Não é necessário estudar toda a biblioteca antes de começar.

## Preparar as primeiras aulas no Windows

Abra a raiz deste workspace no VS Code e use o terminal PowerShell. Confira:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Nesta sessão o comando `python` aponta para Python 3.12.10. Se o seu terminal
apontar para outra instalação, confira o caminho antes de criar o ambiente.
Não é preciso reinstalar Python se a instalação atual atender às dependências.

Execute uma linha por vez:

```powershell
python -m venv .venv-study
.\.venv-study\Scripts\python.exe -m pip install -r requirements-study.txt
.\.venv-study\Scripts\python.exe -m pip check
.\.venv-study\Scripts\python.exe -c "import sys, numpy, cv2, matplotlib; print(sys.executable); print(numpy.__version__, cv2.__version__, matplotlib.__version__)"
```

`venv` cria o ambiente; `-m pip` usa o instalador do Python indicado. O caminho
completo evita depender da ativação do PowerShell. O arquivo de requisitos fixa
as versões presentes na verificação dos notebooks; a instalação em ambiente limpo
ainda precisa ser repetida. Se não houver wheel compatível ou ocorrer conflito,
guarde a mensagem e a versão do Python antes de mudar os requisitos.

Abra [a primeira aula](../notebooks/aulas/01_imagens_e_coordenadas.ipynb).
No canto superior direito, escolha **Select Kernel → Python Environments** e
selecione `.venv-study\Scripts\python.exe`. Execute a preparação e compare o
`sys.executable` mostrado com o caminho do terminal. Salve com Ctrl+S.

Se o ambiente não aparecer, execute **Python: Select Interpreter** na paleta de
comandos e indique esse executável; volte ao seletor de kernel do notebook.
Reinicie o kernel depois de trocar bibliotecas. Não instale pacotes a cada execução
das células: a instalação faz parte da preparação do ambiente.

## Executar o leitor e a preparação já existentes

Esses scripts usam PyYAML, pytest e ZXing-C++, que não são necessários à primeira
aula. Prepare o ambiente próprio na raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.prepare_blanks catalog
.\.venv\Scripts\python.exe -m scripts.prepare_blanks folds
.\.venv\Scripts\python.exe -m scripts.prepare_blanks preview --image 20261112_084042.jpg
.\.venv\Scripts\python.exe scripts/experiment_blanks.py --variants raw clahe otsu
```

Leia `artifacts/catalog/`, a visualização de augmentation, `blanks-folds.json` e
`blanks-report.json`. O [guia de anotações](first_augmentation.md) explica como
interpretar os arquivos. As fotos são entradas; os relatórios derivados ficam em
`artifacts/`. Um status `READ` significa retorno do decoder, sujeito à conferência
independente do conteúdo esperado.

Na leitura deste workspace, o ambiente Python disponível não tinha pytest,
PyYAML nem ZXing-C++; portanto a suíte existente e esse ensaio não foram
reexecutados. As versões históricas do leitor estão no README. A aprovação dos
exemplos do notebook não substitui esses testes.

## Preparar PyTorch quando chegarmos ao carregador

Antes da etapa 04, use o [seletor oficial de instalação](https://pytorch.org/get-started/locally/).
Escolha Windows, Pip, Python e CPU para o primeiro estudo sem pressupor GPU.
Confira a compatibilidade do Python no momento da instalação. CUDA só entrará
se o hardware e o orçamento do experimento justificarem.

Crie outro ambiente:

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install numpy opencv-python PyYAML matplotlib ipykernel pytest
```

O comando de instalação do PyTorch deve ser o fornecido pelo seletor oficial.
Adapte seu início para `.\.venv-training\Scripts\python.exe -m pip install`,
preservando os pacotes e o índice indicados. A escolha acima das outras bibliotecas
é um bootstrap para a próxima etapa, não um ambiente de treino já validado ou
congelado. Não execute treinamento ainda.

Depois de instalar, confira:

```powershell
.\.venv-training\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.zeros(1, 3, 128, 128).shape)"
.\.venv-training\Scripts\python.exe -m pip check
```

O exemplo cria apenas um tensor `[lote, canais, altura, largura]`; não escolhe a
resolução do experimento nem testa a exportação. Quando o carregador estiver
validado, registraremos as versões efetivas num arquivo próprio de treinamento.
Não inclua `torch` nos requisitos do leitor ou da Raspberry Pi.

Siga os tutoriais PyTorch nesta ordem: [tensores](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html),
[Dataset e DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html),
[rede](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html),
[autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
e [otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
Adapte o raciocínio ao nosso alvo espacial; o alvo de classificação do exemplo
FashionMNIST não é o contrato da nossa grade.

## Diagnóstico de erros comuns

| Sinal | O que verificar |
|---|---|
| `ModuleNotFoundError` | Compare o `sys.executable` do notebook com o Python usado em `-m pip` |
| Arquivo não encontrado | Confira `Path.cwd()` e a raiz detectada; abrir o notebook não define necessariamente o diretório atual |
| Cores trocadas | Verifique se `imshow` recebeu RGB e se a leitura OpenCV produziu BGR |
| Caixa deslocada | Confira EXIF, largura/altura, coordenadas XYXY e escalas após arredondamento |
| Exercício `PENDENTE` | Complete a função e execute novamente sua célula e o teste seguinte |
| Exercício `REVISAR` | Leia a mensagem; chame a função de teste diretamente para ver o traceback |
| Foto em caminho com acento não abre no OpenCV | Teste leitura dos bytes com `Path.read_bytes()` e `cv2.imdecode`; esse contorno foi necessário para `Periféricos` nesta sessão |

Os notebooks iniciais trabalham localmente, sem download automático nem chamada
de API. Se futuramente usarmos Colab, será preciso disponibilizar o repositório
e as imagens na sessão e guardar suas alterações ao terminar. Não há configuração
de Colab validada nesta entrega.
