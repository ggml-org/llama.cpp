#!/bin/bash

# إنشاء مجلد build إذا لم يكن موجود
mkdir -p build

# تنزيل الملف التنفيذي server إذا لم يكن موجوداً
if [ ! -f build/server ]; then
  echo "🔽 Downloading server binary..."
  wget -O build/server \
  https://github.com/issa261/llama.cpp/raw/main/build/server
  chmod +x build/server
fi

# إنشاء مجلد models إذا لم يكن موجود
mkdir -p models

# تحميل النموذج إذا لم يكن موجود
if [ ! -f models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf ]; then
  echo "🔽 Downloading model..."
  wget -O models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  https://raw.githubusercontent.com/issa261/github-workflows-download-model.yml/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
fi

# تشغيل الخادم
echo "🚀 Starting server..."
./build/server -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -c 512 -n 256 --host 0.0.0.0 --port 8080
