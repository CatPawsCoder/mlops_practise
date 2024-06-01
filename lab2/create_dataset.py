from sklearn.datasets import load_iris
import pandas as pd

# загрузка данных
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# разделение данных на тренировочный и тестовый наборы
train = data.sample(frac=0.8, random_state=1)  # 80% данных для тренировки
test = data.drop(train.index)  # оставшиеся 20% данных для тестирования

# сохранение датасетов
train.to_csv('data_train.csv', index=False)
test.to_csv('data_test.csv', index=False)