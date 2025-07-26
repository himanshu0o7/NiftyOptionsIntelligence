#!/bin/bash
# Filename: run_webhook_server.sh

echo "🔍 Checking port 8080..."
PID=$(lsof -ti:8080)
if [ -n "$PID" ]; then
  echo "⚠️ Port 8080 in use by PID $PID. Killing..."
  sudo kill -9 $PID
fi

echo "🚀 Starting webhook server..."
python3 webhook_server.py &
sleep 2
echo "🌐 Launching ngrok tunnel..."
ngrok http 8080

