import pickle
import pandas as pd

# загрузка модели
model = pickle.load(open('model.pkl', 'rb'))

# загрузка тестовых данных
data_test = pd.read_csv('data_test.csv')
X_test = data_test.drop(columns=['target']).values

# предсказание для первых пяти экземпляров из тестовой выборки
predictions = model.predict(X_test[:5])
print(f'Predictions for the first five samples: {predictions}')