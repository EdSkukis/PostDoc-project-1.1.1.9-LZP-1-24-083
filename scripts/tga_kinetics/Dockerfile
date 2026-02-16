# Используем легкий образ Python
FROM python:3.10-slim

# Устанавливаем рабочую папку
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в папку /app
COPY . .

# Создаем папки для данных, если их нет
RUN mkdir -p data_csv kinetics_results data_modified

# Открываем порт 8000
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]