import pandas as pd
from catboost.datasets import titanic

# Загрузка датасета
train, _ = titanic()

# Выбор необходимых колонок
dataset = train[['Pclass', 'Sex', 'Age']]

# Сохранение датасета
dataset.to_csv('titanic_dataset.csv', index=False)