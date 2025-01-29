# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'model.joblib')

# Оценка модели
print("Model accuracy:", model.score(X_test, y_test))
