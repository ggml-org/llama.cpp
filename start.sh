#!/bin/bash

# إنشاء مجلد النماذج إذا لم يكن موجود
mkdir -p models

# تحميل النموذج إذا لم يكن موجود
if [ ! -f models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf ]; then
  echo "Downloading model..."
  wget -O models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  https://raw.githubusercontent.com/issa261/github-workflows-download-model.yml/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
fi

# تشغيل السيرفر من مجلد build
./build/server -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -c 512 -n 256 --host 0.0.0.0 --port 8080
