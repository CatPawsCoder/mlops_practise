# Шаг 1: Создание Python-скрипта для генерации данных (data_creation.py)
import numpy as np
import pandas as pd
import os

def create_data(num_samples=100, noise=False, anomalies=False):
    np.random.seed(10)
    
    # Generate synthetic data
    time = np.arange(0, num_samples)
    # Добавить колебания температуры и шум
    temperature = 20 + 10 * np.sin(time / 5) + (np.random.randn(num_samples) * (0.5 if noise else 0))

    # Introduce anomalies
    if anomalies:
        # кол0во аномалий которые будут добавлены в данные 
        #  5% от общего количества данных (num_samples). Если, например, у вас 1000 образцов данных, то 5% от 1000 это 50.
        # int(...): Приводит результат к целому числу. Это нужно, потому что количество аномалий должно быть целым числом.
        num_anomalies = int(0.05 * num_samples)
        # ыбирает случайные индексы в массиве данных, в которые будут добавлены аномалии.
        # Разбор выражения:
        # np.random.choice(num_samples, num_anomalies, replace=False): Выбирает num_anomalies уникальных случайных индексов из диапазона от 0 до num_samples - 1.
        anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
        # Вносит аномалии в данные, добавляя или вычитая 10 градусов в выбранных индексах.
        temperature[anomaly_indices] += np.random.choice([-10, 10], num_anomalies)
    
    data = pd.DataFrame({'time': time, 'temperature': temperature})
    return data

def save_data(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(os.path.join(path, filename), index=False)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    train_data = create_data(num_samples=1000, noise=True, anomalies=True)
    test_data = create_data(num_samples=300, noise=True, anomalies=True)
    
    save_data(train_data, os.path.join(base_path, 'train'), 'train_data.csv')
    save_data(test_data, os.path.join(base_path, 'test'), 'test_data.csv')