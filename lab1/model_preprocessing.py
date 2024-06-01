import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    scaler = StandardScaler()
    data[['temperature']] = scaler.fit_transform(data[['temperature']])
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    
    preprocess_data(os.path.join(project_path, 'train', 'train_data.csv'), os.path.join(project_path, 'train', 'train_data_scaled.csv'))
    preprocess_data(os.path.join(project_path, 'test', 'test_data.csv'), os.path.join(project_path, 'test', 'test_data_scaled.csv'))