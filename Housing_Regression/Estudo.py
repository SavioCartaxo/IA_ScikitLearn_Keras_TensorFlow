#!/usr/bin/env python
# coding: utf-8

# In[68]:


# Importado as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from  zlib import crc32
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedShuffleSplit


# In[69]:


# lendo o csv
preco_das_casas_df = pd.read_csv("housing.csv")


# In[70]:


# Observando o arquivo csv
preco_das_casas_df.head()


# In[71]:


# Informações da tabela
preco_das_casas_df.info()


# In[72]:


# Observei que a coluna ocean_proximity é dividido em categorias, quais são essas categorias?
preco_das_casas_df["ocean_proximity"].value_counts()


# In[73]:


preco_das_casas_df.hist(bins=50, figsize= (20,50))
plt.show()


# In[74]:


def dados_de_teste(dados, razao):
    ## Função com 2 retornos
    ## O primeiro: um Serial do Pandas, com os itens do treino da IA
    ## O Segundo   um Serial do Pandas, com os itens não escolhidos para o treinamento, pois serão testados para validar
    # o Algoritmo

    indices_embaralhados = np.random.permutation(len(dados))
    quantidade_de_dados_para_treino = np.trunc(len(dados) * razao)
    itens_do_treino = indices_embaralhados[:quantidade_de_dados_para_treino]
    teste_indices = indices_embaralhados[quantidade_de_dados_para_treino:]
    return dados.iloc[itens_do_treino], dados.iloc[teste_indices]


# A Função de cima têm um problema:
#     toda vez que o código for instanciado, ele gerará novos valores, o que é ruim e cria algoritmos diferentes quando o código for re instanciado
# 
# Como resolver isso?
# 
#     salvar os dados que serão usados não será o suficiente, pois quando forem introduzidos mais casos de teste, o algoritmo terá de ser rebobinado, modificando, mais uma vez os casos que alimentarão o algoritmo
# 
# A Solução: criar uma espéie de "Marcador" que define se um caso é usado para testes ou se é usado para alimentar o algritmo, com a função crc32 da bibliotca zlib

# In[75]:


def identificador_de_testes(id_, proporcao_testes) :
    # id_ é um identificador único que marca as células como testes
    # proporcao_testes é a porcentagem(ex.: 0.2 = 20%) que vocÊ quer separar para os testes(ex.: 0.2 = 20%)
    #
    # Retorna uma expressão booleanda(True se a porcentagem de células marcadas forem menores que a proporcao_testes)

    return crc32(np.int64(id_)) & 0xffffffff < proporcao_testes * 2**32


# In[76]:


def divide_testes_de_treino_por_id(dados, proporcao_testes, nome_da_coluna):
    # dados = DataFrame com todos os dados
    # proporcao_testes = porcetagem desejada de casas separads para trino
    # nome_da_coluna = o nome da cplluna criada para o identficador
    #
    # 2 retornos
    # linhas selecionadas para treino
    # linhas selecionadas para teste

    ids = dados[nome_da_coluna]
    esta_no_teste = ids.apply(lambda id_: identificador_de_testes(id_, proporcao_testes))

    return dados.loc[~esta_no_teste], dados.loc[esta_no_teste]


# Como o Data Frame não tem uma coluna que marque as linhas como treinamento ou teste, é necessário cria-la
#     uma solução boa para esse problema é definir a coluna dos indeces como marcadores, da seguinte forma:

# In[77]:


casas_com_id = preco_das_casas_df.reset_index()
dados_treino, dados_teste = divide_testes_de_treino_por_id(casas_com_id, 0.2, "index")


# O Scikitlear já possui uma função que faz isso, que é:
# 
#     from sklear.model_selection import train_test_split
# 
# Parâmetros: 
# 
#     train_test_split(DataFrame, test_size = 0.2, random_state = 42)
# 
#     random_state = é uma seed, se setar uma seed como um número qualquer, os selecionados para teste e treino serão sempre os mesmos

# In[78]:


dados_treino, dados_teste = train_test_split(preco_das_casas_df, test_size= 0.2, random_state=42)


# Geralmente, é necessário dividir os dados em grupos antes de escolher (a mesma porcentagem) de amostras de cada grupo, como por exmplo, divir o grupo entre homens e mulheres e escolher 50% de cada grupo para evitar o VIÉS DE AMOSTRAGEM
# 
# como fazer isso? 
#     Usando a função cut do Pandas
# 
# O exemplo a seguir divide o grupo das casas de acordo com a renda de cada casa, dividindo-as em 5 categorias

# In[79]:


# Função cut do Pandas, recebe os parametros scopo, bins e labels
preco_das_casas_df["income_cat"] = pd.cut(preco_das_casas_df["median_income"], 
                                          bins = [0. , 1.5 , 3. , 4.5 , 6., np.inf],
                                          labels = [1, 2, 3, 4, 5])

preco_das_casas_df["income_cat"].hist() # Criou a coluna income_cat com essas caterias


# Agora sim, podemos fazer uma seleção de dados mais bem distribuida, e para isso, usaremos:

# From sklear.model_selection import StratifiedShuffleSplit

# In[80]:


dividido = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for indice_treino, indice_teste in dividido.split(preco_das_casas_df, preco_das_casas_df["income_cat"]):
    df_treino = preco_das_casas_df.loc[indice_treino]
    df_teste  = preco_das_casas_df.loc[indice_teste]


# In[81]:


#Removendo a coluna de estratificação, pois não será mais útil para nós
preco_das_casas_df = preco_das_casas_df.drop("income_cat", axis=1)

# Criar uma cópia dos dados para usar por enquanto
copia_df = preco_das_casas_df


# In[82]:


copia_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4) # Vizualização melhor


# Muito bom, mas vamos mexer um pouco mais aqui, para deixar mais visivel, faremos o seguinte:
# 
# 1 - Mudar o tamanho do circulo de acordo com a população da áres
# 
# 2 - Mudar a cor de acordo com o preço (vermelho = mais caro)

# In[83]:


copia_df.plot(kind="scatter", x="longitude", y="latitude",

              s=copia_df["population"] / 100, label="population", figsize=(10, 7), # Tamanho 
              c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)   # Preço média


# PREPARANDO OS DADOS PARA O APRENDIZADO DE MÁQUINA
# 
# O ScikitLear tem uma Classe que ajuda a tratar dados vazios(o que pode reduzir o uso da função fillna no código),
# a função é a seguinte:
# 
#     from sklear.impute import SimpleImputer
# 
# podemos usar isso para preencher todas as colunas de dados numericos vazios com a média de suas colunas, por exemplo

# In[28]:


# teste e treino
from sklearn.model_selection import StratifiedShuffleSplit

divisao = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for treino_index, teste_index in divisao.split(preco_das_casas_df, preco_das_casas_df["ocean_proximity"]):
    treino = preco_das_casas_df.iloc[treino_index]
    teste  = preco_das_casas_df.iloc[teste_index]


# In[86]:


# Preenchendo valores ausentes

from sklearn.impute import SimpleImputer

casas_numeros = preco_das_casas_df.drop(["ocean_proximity"], axis=1)

completa = SimpleImputer(strategy="mean")
completa.fit(casas_numeros)
completa = completa.transform(casas_numeros)

casas = pd.DataFrame(completa, columns=casas_numeros.columns, index=casas_numeros.index)
casas = pd.merge(casas, preco_das_casas_df["ocean_proximity"], left_index=True, right_index=True)

display(casas)


# Nesse caso, apenas o atributo total_badrooms tem células vazias, mas, se forem adicionados mais dados, alguns deles podem vir, também vazios
# 
# sendo assim, usamos o imputer para lidar com dados futuros

# ETAPA FUNDAMENTAL
# 
# precisamos decidir o que fazer com os dados não preenchidos
# 
# não usaremos o pd.filla, pois ele é lento(em digitação) e não leva em consideração os dados que serão adicionados posteriormente.
# 
# então, o que usaremos?
# 
#     from sklearn.impute import SimpleImputer
# 
# o argumento que ele usa é strategy, que é a métrica que ele usa para completar os argumentos
# 
# aqui, utilizarei o argumento mean

# In[ ]:


###########################################################################################################################################################################


# Como tratar Strings e normalizar os dados ao mesmo tempo?
# 
#     from sklearn.compose import ColumnTransformer
# 
# isso é uma classe que consegue tratar os dados, todos ao mesmo tempo.
# 
# para isso, seguir o passo a passo:
# 
# 1 - separar o df em colunas numéricas e categoricas
# 
# 2 - instanciar objeto SolumnTransformer
# 
#     full_pipeline = ColumnsTransformer([
#         ("num", num_pipeline, *inserir dados numericos*),
#         ("cat", oneHotEncoder(), *inserir dados categoricos*)
#     ])
# 
# 3 - criar variavel dos dados tratados
# 
#     dados = full_pipeline.fit_transform(dados_iniciais)

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

casas_numeros 
casa_categorias = preco_das_casas_df["ocean_proximity"]

tudo_tratado = ColumnTransformer ([
    ("num", num_pepline) # Precisa do StandardScatter antes
]) 

