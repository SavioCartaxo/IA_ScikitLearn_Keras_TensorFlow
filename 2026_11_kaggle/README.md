O solved_01 deu certo, fiquei com mais de 94% de acuracia.
O solved_02 aumento para 0.94265.
O solved_03 aumentou para  0.94353.
 -> usei 95% dos dados para treino
O solved_04 (../estudos/classes_desbalanceadas.ipynb) diminuiu para 0.93867
 -> usei oversample com SMOTE
O solved_05 foi para 0.94174
 -> não usei todos os dados para treino, então tudo bem, ta melhorando.
 -> oq fiz? Feature engenier
 -> df['proporcao_gasta_na_transacao'] = df['amount_usd'] / df['account_balance_usd']
 -> df = df.drop(columns=['amount_usd', 'account_balance_usd'])
O solved_06 foi para 0.94311
 -> Feature engenier com outras 2 classes
O solved_07 melhorou para 0.94537
 -> ajuste das metricas