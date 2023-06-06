==========================================
Title:  Classificação de textos jurídicos com segmentação semântica
Author: Danilo Panzeri Carlotti
Date:   06 de Junho de 2023
==========================================

# INSTRUÇÕES DO PACOTE DE CLASSIFICAÇÃO COM SEGMENTAÇÃO SEMÂNTICA

Scripts contidos neste pacote:

- Treinamento e visualização de métricas associadas a classificadores binários que podem ser treinados a qualquer tempo de maneira independente uns dos outros
- Segmentação semântica de textos de decisões judiciais (acórdãos ou sentenças)
- Classificação de acórdãos ou decisões desconhecidas tendo como output uma tabela ou arquivo (csv ou excel) com as probabilidades de pertencimento de todos os textos a cada uma das possíveis classes

### INPUT

Arquivo csv contendo as colunas:
- codigos_movimentos_temas: classe da decisão (sentença ou acórdão)
- conteudo: coluna contendo o texto em si da decisão (sentença ou acórdão)

Arquivo .env contendo:
- as possíveis classes que aparecerão no arquivo csv que servirá como input para treinamento dos classificadores

### OUTPUT

- Modelos binários de classificação treinados em todas as classes selecionadas
- Classificação de textos selecionados pelo usuário indicando, para cada texto, a probabilidade de pertencimento dos textos a cada uma das possíveis classes existentes

## PASSO A PASSO DA SEGMENTAÇÃO DE TEXTOS

Se necessário, fazer ajustes no código do script `main_segmentation.py` para alterar parâmetros como nome do arquivo ou caminho. 
Quando pronto, rodar:

`cd scripts`
`python main_segmentation.py`

## PASSO A PASSO DO TREINAMENTO

Para treinar os modelos é necessário rodar

`cd scripts`
`python prepare_data_classification.py`
`python train_model.py`

Em caso de testes com o uso de oversampling, deve-se alterar o parâmetro `to_oversample = True` da função `train_model`

O output da função `plot_calibration` é um gráfico com o resultado da calibração dos modelos.
O output da função `train_model` são os modelos binários que serão salvos na pasta `models`

## PASSO A PASSO PARA VALIDAÇÃO DOS MODELOS

Para validar o desempenho dos modelos em um outro dataset específico, execute:

- Defina um valor para a variável `PATH_FILES_VALIDATE_MODELS`
- Rode a função `main_classification_multiple_texts`, comentando o resto
- `cd scripts`
- `python classification.py`

O output será um arquivo csv salvo no caminho especificado com as classes e se cada modelo acertou ou não o palpite, usando como threshold o valor padrão de 0.5 ou 50%, que é o valor padrão com o qual os modelos são treinados, salvo refatoração do código em `train_model.py`

## PASSO A PASSO DA CLASSIFICAÇÃO DE TEXTOS DESCONHECIDOS

Para classificar dados tendo em vista os modelos existentes, execute:

- Defina um valor para a variável `PATH_FILES_CLASSIFY`
- Rode a função `main_classification_multiple_texts_not_classified`, comentando o resto
- `cd scripts`
- `python classification.py`

O output será um arquivo csv salvo no caminho especificado com uma linha associada a cada um dos textos e nas colunas, cada um dos modelos e suas respectivas probabilidades de um texto específico pertencer a alguma daquelas classes


## PASSO A PASSO PARA MUDANÇA DA SEGMENTAÇÃO POR FATOS PARA DECISÃO OU QUALQUER OUTRA CLASSE

Necessário mudar as seguintes linhas nos seguintes arquivos:
- Linhas 14 e 38, prepare_data_classification.py
- Linha 45 train_model.py
- Linhas 47, 72, 86 e 89 do classification.py
