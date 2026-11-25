# Aulas e laboratórios do PAIC

Comece pela [preparação do ambiente](../docs/tools_setup.md) e acompanhe a
[trilha completa](../docs/study_workflow.md).

| Assunto disponível | Aula guiada | Exercícios |
|---|---|---|
| Imagens, cores, ROI, centro e transformações | [Aula 01](aulas/01_imagens_e_coordenadas.ipynb) | [Laboratório 01](laboratorios/01_imagens_e_coordenadas.ipynb) |

Execute uma célula por vez e escreva sua previsão antes de olhar a saída.
No laboratório, substitua `NotImplementedError` por sua implementação. Os testes
mostram `PENDENTE`, `REVISAR` ou `APROVADO`; eles não importam a solução da aula.
Após uma alteração, execute novamente a definição da função e o teste seguinte.
As perguntas de interpretação são parte da atividade.

Os notebooks são entregues sem saídas e sem execuções salvas. Salve suas tentativas
com Ctrl+S ou faça uma cópia com outro nome nesta pasta. As células fornecidas
não sobrescrevem imagens, não exportam relatórios e não executam treinamento.
Exemplos geométricos usam dados sintéticos; a aula também abre uma única foto
real para examinar forma, memória e orientação. Essa exibição não decodifica códigos.

## Verificação desta entrega

Python 3.12.10, NumPy 2.5.2, OpenCV 5.0.0.93, Matplotlib 3.11.1,
nbformat 5.11.1 e nbclient 0.11.0 estavam disponíveis no PC. Os notebooks foram
validados como documentos e suas células executadas em ordem, em kernel novo.
O laboratório vazio mantém os exercícios pendentes; soluções temporárias foram
injetadas somente na cópia de verificação para testar os casos corretos. Também
foi conferido que uma implementação incorreta aparece como `REVISAR`.

As cópias executadas e o relatório técnico ficam em
`artifacts/study-validation/`, ignorados pelo Git. A instalação limpa dos requisitos,
a interface gráfica do VS Code, o leitor ZXing-C++, o PyTorch e a Pi 3 não foram
validados nesta entrega. As versões do leitor continuam em seu arquivo próprio.
