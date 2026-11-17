
# XG boost

Um empilhamento de modelos fracos. A saida é uma media das saidas dos modelos fracos.

### Casos de Uso

O reforço de gradiente fornece um bom equilíbrio de precisão, eficiência e escalabilidade que pode ser amplamente aplicado a:

<ul>
    <li>Classificação: Prever categorias ou classes (por exemplo, detecção de spam, detecção de fraude)</li>
    <li>Regressão: Prever valores numéricos (por exemplo, previsão de preços de ações, previsão de vendas).</li>
    <li>Ranking: Classificação de itens com base em sua relevância ou importância (por exemplo, resultados de pesquisa, recomendações)</li>
</ul>

### Gradient boost
<ul>
    <li>[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier)</li>
    <li>Classification Analysis with XGBoost : https://www.kaggle.com/code/tirendazacademy/classification-analysis-with-xgboost</li>
    <li>O HistGradientBoostingClassifier é uma versão muito mais rápida desse algoritmo para conjuntos de dados médios e grandes (n_samples >= 10.000) e oferece suporte a restrições de monotonicidade.</li>
<ul>



RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html