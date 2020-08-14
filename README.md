
# Multilayer Perceptron em Python

Rede Neural - com uma camada oculta e uma camada de saída - treinada com o método **Mini-Batch Gradient Descent** para retropropagação do erro.
(O valor médio da função de custo após o processamento de cada *mini-batch* é plotado em um gráfico com eixos *Mini-batch x Average cost*)

## Diretório 'files'

Contém dois arquivos:

- **iris.csv**
	 Dataset de treinamento com um conjunto de entradas (cada uma composta por 4 números) e saídas esperadas (cada uma composta por 3 números) para a correta classificação das variações da flor Iris em uma das 3 espécies existentes e representadas no referido arquivo pelas tuplas **(1, 0, 0)**, **(0, 1, 0)** e **(0, 0, 1)**
- **testResults.csv**
	Saídas obtidas/geradas pela rede neural (após a retropropagação do erro e atualização dos parâmetros) e correspondentes às entradas do dataset (podem ser comparadas às saídas esperadas para análise da taxa de acerto da classificação realizada pela MLP)

## Bibliotecas utilizadas
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

