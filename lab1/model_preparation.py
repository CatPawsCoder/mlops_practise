import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import joblib

def train_model(train_data_path, model_path):
    data = pd.read_csv(train_data_path)
    X = data[['time']]
    y = data['temperature']
    
    model = LinearRegression()
    model.fit(X, y)
    
    joblib.dump(model, model_path)

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    
    train_model(os.path.join(project_path, 'train', 'train_data_scaled.csv'), os.path.join(project_path, 'train', 'model.joblib'))