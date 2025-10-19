#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# Lendo e analisando os dados

# In[2]:


casas_df = pd.read_csv("housing.csv")
display(casas_df.head(5))
casas_df.info()


# # TRATANDO OS DADOS

# In[3]:


# CRIANDO CATEGORIAS

# Criando coluna "Pessoas por casa"
casas_df["person per house"] = (casas_df["population"] / casas_df["households"]).round(3)
# Tratando alguns erros dessa coluna
casas_df.loc[casas_df["person per house"] > 14] = casas_df["person per house"].mean()

# Criando a coluna "Quartos por casa"
casas_df["bedrooms per house"] = (casas_df["total_bedrooms"] / casas_df["households"]).round(3)
# Tratando alguns problemas
casas_df = casas_df[casas_df["bedrooms per house"] <= 10]

# Criando a coluna "Cômodos por casa"
casas_df["rooms per house"] = (casas_df["total_rooms"] / casas_df["households"]).round(3)

# Tratado um erro da coluna "ocean_proximity"
casas_df = casas_df[casas_df["ocean_proximity"].isin(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])]


# In[4]:


# Criando os grupos sociais para estretificar(dividido em classe baixa, classe média e classe alta)

# Organizando por ordem 
casas_organizadas_por_renda = casas_df["median_income"].sort_values().reset_index(drop=True)

# Definindo o índice do ultimo participante da classe baixa e media
renda_base = np.min(casas_df["median_income"])
um_porcento_da_renda = (np.max(casas_df["median_income"]) - np.min(casas_df["median_income"])) / 100

classe_baixa_limite = renda_base + 28 * um_porcento_da_renda # 4.559956
classe_media_limite = renda_base + 81 * um_porcento_da_renda # 12.245061999999999

indice_ultima_casa_classe_baixa = np.argmax(casas_organizadas_por_renda >= classe_baixa_limite).astype(int) + 1
indice_ultima_casa_classe_media = np.argmax(casas_organizadas_por_renda >= classe_media_limite).astype(int) + 1

casas_df["class"] = pd.cut(
    casas_df["median_income"],
    bins = [
        0,
        casas_organizadas_por_renda[indice_ultima_casa_classe_baixa],
        casas_organizadas_por_renda[indice_ultima_casa_classe_media],
        np.inf
        ],
    labels = ["Lower class", "Middle class", "Upper class"]
)


# # Observando as correlações

# In[5]:


display(casas_df[["longitude",
          "latitude",
          "housing_median_age", 
          "total_rooms", 
          "total_bedrooms", 
          "population", 
          "households", 
          "median_income", 
          "median_house_value", 
          "person per house", 
          "bedrooms per house", 
          "rooms per house"]
          ].corr().round(3)
)

valor_das_casas_de_acordo_com_a_proximidade_do_oceano = casas_df.groupby("ocean_proximity")["median_house_value"].mean().round(2).reset_index()
valor_das_casas_de_acordo_com_a_proximidade_do_oceano = valor_das_casas_de_acordo_com_a_proximidade_do_oceano.sort_values("median_house_value", ascending=False)
display(valor_das_casas_de_acordo_com_a_proximidade_do_oceano)


# # Resultado da análise:
# 
# Muitos dados não estão correlacionados com os preços das casas(o que é ruim)
# 
# Um dos principais features relacionados ao valor da casa é a media da renda, Pessoas que ganham mais tendem a viverem
# em casas mais caras
# 
# observe a ordem de proximidade com o oceano:
#  
#  1 - ISLAND
#  
#  2 - NEAR OCEAN
#  
#  3 - NEAR BAY	
# 
#  4 - <1H OCEAN	
# 
#  5 - INLAND	
# 
# Oberva-se que, no geral, o preço das casas tende a aumentar de acordo com a proximidade do Oceano

# In[6]:


# Separando os dados treino e teste

divisao = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for treino_index, teste_index in divisao.split(casas_df, casas_df["class"]):
    dados_treino = casas_df.iloc[treino_index] # 16.324 itens
    dados_teste  = casas_df.iloc[teste_index ] #  4.081 itens

