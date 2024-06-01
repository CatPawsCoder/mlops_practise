@echo off

echo "----Create Dataset (begin)-----"
%PYTHON_PATH%\python.exe create_dataset.py
echo "----Create Dataset (end)-----"

echo "----Train the Model (begin)-----"
%PYTHON_PATH%\python.exe train_model.py
echo "----Train the Model (end)-----"

echo "----Use the Model for Prediction (begin)-----"
%PYTHON_PATH%\python.exe make_prediction.py
echo "----Use the Model for Prediction (end)-----"