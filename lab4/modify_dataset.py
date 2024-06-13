import pandas as pd

# Загрузка датасета
dataset = pd.read_csv('titanic_dataset.csv')

# Создаем новый столбец с заполненными значениями
dataset['Age_filled'] = dataset['Age'].fillna(dataset['Age'].mean())

# Сохраняем измененный датасет
dataset.to_csv('titanic_dataset_v2.csv', index=False)
