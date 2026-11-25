# Registro de experimento

Copie este modelo para um registro com nome próprio. Exemplos sintéticos devem
ser identificados como sintéticos; os campos abaixo não têm valores de produção
preenchidos. Guarde relatórios detalhados em `artifacts/` e preserve neste registro
as condições e conclusões necessárias para reproduzir a comparação.

## Pergunta e previsão

- Pergunta que o experimento responde:
- Resultado previsto e motivo:
- Única variável alterada, ou fatores explicitamente planejados:
- Comparação de referência:

## Condições

- Data, autor e revisão do código:
- Comando ou notebook e ordem de execução:
- Sistema operacional, executável Python e versões das bibliotecas:
- Fonte dos dados, manifesto, revisão e critério das anotações:
- Peças/faces/sessões por divisão:
- Inclusão ou exclusão de `edge_contact`:
- Semente, transformações, tamanho da entrada, grade e normalização:
- Orçamento de execuções, épocas e critérios de parada, quando houver treino:
- Arquivos de configuração e caminho dos resultados:

## Verificações

- Formatos, eixos, tipos e unidades conferidos:
- Ausência de peça compartilhada entre treino e validação:
- Visualizações de imagem, caixa, centro e grade conferidas:
- Casos esperados de falha e respectivos estados:
- Testes executados, versões e resultados:

## Resultado e interpretação

| Caso ou divisão | Condição de referência | Condição alterada | Métrica com unidade | Observação |
|---|---|---|---|---|
| Preencher | Preencher | Preencher | Preencher | Preencher |

- O que os dados sustentam:
- O que pode explicar a diferença:
- O que este experimento não permite concluir:
- Decisão e condição para revisá-la:
- Próxima pergunta:

## Campos adicionais para bancada e esteira

- Modelo exato da Pi, sistema, runtime, formato e quantização:
- Câmera, resolução original, iluminação, exposição e foco:
- Velocidade e espaçamento medidos; sensor ou método de passagem:
- Origem independente do conteúdo esperado do Data Matrix:
- Quantidade de passagens, leituras corretas, falhas, perdas e duplicidades:
- Captura, pré-processamento, inferência, ROI, leitura e total por passagem:
- p50/p95, vazão, RAM máxima e CPU, com método de medição:
- Entregas pendentes/enviadas e teste de indisponibilidade da API:

Use contagens e denominadores explícitos. Não inclua o conteúdo real dos códigos,
tokens ou URLs privadas no registro versionado.
