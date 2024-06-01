import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import joblib

def test_model(test_data_path, model_path):
    data = pd.read_csv(test_data_path)
    X = data[['time']]
    y_true = data['temperature']
    
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    
    test_model(os.path.join(project_path, 'test', 'test_data_scaled.csv'), os.path.join(project_path, 'train', 'model.joblib'))