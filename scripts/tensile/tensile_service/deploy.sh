#!/bin/bash
# High-level deployment script for Tensile Service

SERVER_IP="62.3.175.167"
PORT="8080"
USER="ubuntu"
REMOTE_PATH="~/tensile_service"
SSH_KEY="~/.ssh/id_ed25519"

echo "Step 1: Creating remote directories..."
ssh -i $SSH_KEY $USER@$SERVER_IP "mkdir -p $REMOTE_PATH/{core,preprocessing,app,storage}"

echo "Step 2: Copying EVERYTHING (including maintenance.py)..."
# rsync копирует всё дерево проекта одним махом
# --exclude исключает то, что не нужно на сервере
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude 'storage' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.env' \
    ./ $USER@$SERVER_IP:$REMOTE_PATH/

echo "Step 3: Building and launching containers..."
ssh -i $SSH_KEY $USER@$SERVER_IP "cd $REMOTE_PATH && docker compose up -d --build"

# ИСПРАВЛЕННЫЙ CRON (используем правильные переменные и запуск внутри контейнера)
echo "Step 4: Setting up Cron job..."
CRON_CMD="0 0 * * * cd $REMOTE_PATH && docker compose exec -T tensile_api python3 maintenance.py"
ssh -i $SSH_KEY $USER@$SERVER_IP "(crontab -l 2>/dev/null; echo \"$CRON_CMD\") | sort -u | crontab -"

echo "Deployment finished. API: http://$SERVER_IP:$PORT/docs"