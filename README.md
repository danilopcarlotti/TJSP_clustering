# CLUSTERIZAÇÃO DE DOCUMENTOS COM GERAÇÃO DE RELATÓRIO DOS DOCUMENTOS REPRESENTATIVOS

### PARA GERAR UM RELATÓRIO DOS TEXTOS REPRESENTATIVOS DE UM GRUPO DE TEXTOS

- Salvar um arquivo com os textos em uma tabela (csv, xlsx ou parquet) com uma coluna chamada "text"
- Variáveis: 
    - path_file : str = local do arquivo
    - type_file : str = csv, xlsx ou parquet
    - just_final_decisions : int = 1, caso queira somente decisões finais (sentenças e acórdãos), 0, caso contrário
    - output_path : str = local onde o arquivo "relatório.docx" será salvo

- Para geração do relatório rodar os comandos:

    `cd scripts`
    `python main.py [path_file] [type_file] [just_final_decisions]`


### PARA COMPARAR TEXTOS DE PETIÇÕES COM TEXTOS DE DECISÕES

- Salvar um arquivo com os textos das decisões e das petições selecionados. Os textos devem estar sob uma coluna com o nome "text". Deve haver mais uma coluna denominada "class". Cada texto de decisão deve constar nessa coluna com o valor "decisão" e cada texto de petição deve constar nessa coluna com o valor "petição".
- Para geração da comparação dos textos em similaridade, rodar os comandos:

    `cd scripts`
    `python sim_arrays.py [path_file] [type_file]`
