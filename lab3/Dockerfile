# Используем официальный Python образ
FROM python:3.8-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для API
EXPOSE 5000

# Запуск Flask приложения
CMD ["python", "app.py"]
