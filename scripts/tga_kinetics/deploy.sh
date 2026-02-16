#!/bin/bash

# Конфигурация
SERVER_IP="62.3.175.167"
USER="ubuntu"
KEY="~/.ssh/id_ed25519"
REMOTE_DIR="~/tga_kinetics"

echo "Starting to deploy the project on $SERVER_IP..."

# 1. Создаем структуру папок на сервере
echo "Create folders on the server..."
ssh -i $KEY $USER@$SERVER_IP "mkdir -p $REMOTE_DIR/methods $REMOTE_DIR/preprocessing $REMOTE_DIR/data_csv $REMOTE_DIR/kinetics_results $REMOTE_DIR/data_modified"

# 2. Копируем логику (папки)
echo "Copying modules..."
scp -i $KEY -r ./methods/* $USER@$SERVER_IP:$REMOTE_DIR/methods/
scp -i $KEY -r ./preprocessing/* $USER@$SERVER_IP:$REMOTE_DIR/preprocessing/

# 3. Копируем файлы конфигурации
echo "Copying configuration files..."
scp -i $KEY ./main.py $USER@$SERVER_IP:$REMOTE_DIR/
scp -i $KEY ./Dockerfile $USER@$SERVER_IP:$REMOTE_DIR/
scp -i $KEY ./docker-compose.yml $USER@$SERVER_IP:$REMOTE_DIR/
scp -i $KEY ./requirements.txt $USER@$SERVER_IP:$REMOTE_DIR/

echo "🛠 Restarting Docker container on server..."
ssh -i $KEY $USER@$SERVER_IP "cd $REMOTE_DIR && docker compose down && docker compose up -d --build"

echo "⏳ Waiting for launch API (5 сек)..."
sleep 5

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://$SERVER_IP:8000/health)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ Deployment successful! API is working (Status 200)."
    echo "🌍 Swagger UI: http://$SERVER_IP:8000/docs"
else
    echo "❌ ERROR: The server responded with a status $HTTP_STATUS or unavailable."
    echo "📝 Check the logs with the command: ssh -i $KEY $USER@$SERVER_IP 'docker logs tga-container'"
fi