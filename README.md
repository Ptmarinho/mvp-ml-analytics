# MVP — Machine Learning & Analytics

> Previsão de salários de profissionais de IA e Dados (regressão supervisionada)

**Autora:** Patricia Marinho

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ptmarinho/mvp-ml-analytics/blob/main/mvp_salarios_ia.ipynb)

## Conteúdo do repositório

- `mvp_salarios_ia.ipynb` — notebook único do MVP (executável de ponta a ponta no Google Colab)
- `ai_ml_salaries.csv` — dataset (pesquisa salarial pública ai-jobs.net, 73.148 registros, 2020–2025)

## Como executar

1. Abra o notebook no Google Colab.
2. Runtime → Run all. Nenhuma configuração, upload, login ou chave de API é necessária.
3. O dataset é carregado automaticamente via URL raw deste repositório.

Tempo total de execução: ~1 minuto em CPU padrão do Colab.

## Resumo da solução

Baseline (mediana) + Ridge + Random Forest (bagging) + HistGradientBoosting (boosting) + ensemble por
votação, comparados por validação cruzada 5-fold; HistGradientBoosting otimizado via RandomizedSearchCV.
Avaliação com MAE, MedAE, RMSE, RMSLE, R², MAPE e acurácia por tolerância, além de curva de aprendizado
e ablação de atributos. Resultado no teste: MAE ≈ US$ 48 mil, R² ≈ 0,20, contra MAE ≈ US$ 58 mil do
baseline — com discussão crítica do teto de desempenho imposto pelos atributos disponíveis.
