import pandas as pd

# Загрузка датасета
dataset = pd.read_csv('titanic_dataset_v2.csv')

# One-hot encoding для поля 'Sex'
dataset = pd.get_dummies(dataset, columns=['Sex'], drop_first=True)

# Сохранение датасета с новым признаком
dataset.to_csv('titanic_dataset_v3.csv', index=False)
