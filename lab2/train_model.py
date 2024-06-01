import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# загрузка данных
data_train = pd.read_csv('data_train.csv')
X_train = data_train.drop(columns=['target']).values
y_train = data_train['target'].values

# обучение модели
model = LogisticRegression(max_iter=100_000).fit(X_train, y_train)

# сохранение модели
pickle.dump(model, open('model.pkl', 'wb'))