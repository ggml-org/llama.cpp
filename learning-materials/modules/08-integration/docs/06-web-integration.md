# Web Integration - JavaScript and WebAssembly

**Module 8, Lesson 6**
**Estimated Time**: 3 hours
**Difficulty**: Intermediate to Advanced

## Overview

Run llama.cpp directly in web browsers using WebAssembly (WASM), enabling client-side AI inference without servers. Learn JavaScript bindings, browser optimization, and building web-based AI applications.

## Learning Objectives

- Compile llama.cpp to WebAssembly
- Use JavaScript bindings for llama.cpp
- Build browser-based chat applications
- Optimize for web performance
- Handle large models in browsers

## Prerequisites

- JavaScript/TypeScript knowledge
- Understanding of async programming
- Web development basics (HTML/CSS)
- Module 8, Lessons 1-3

---

## 1. WebAssembly Compilation

### Building llama.cpp for WASM

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with Emscripten
mkdir build-wasm
cd build-wasm

emcmake cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

emmake make -j

# This creates llama.wasm and llama.js
```

### Custom Build Script

```bash
#!/bin/bash
# build-wasm.sh - Custom WASM build script

set -e

EMCC_FLAGS=(
  -O3
  -s WASM=1
  -s ALLOW_MEMORY_GROWTH=1
  -s MAXIMUM_MEMORY=4GB
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_llama_init","_llama_generate"]'
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]'
  -s MODULARIZE=1
  -s EXPORT_NAME="createLlamaModule"
  -s ENVIRONMENT='web,worker'
  -s SINGLE_FILE=1
)

# Compile
emcc \
  "${EMCC_FLAGS[@]}" \
  -I./include \
  -I./common \
  src/llama.cpp \
  src/ggml.c \
  -o llama.js

echo "Build complete: llama.js"
```

---

## 2. JavaScript Bindings

### Loading WASM Module

```javascript
// llama-wasm.js
class LlamaWasm {
  constructor() {
    this.module = null;
    this.context = null;
  }

  async initialize(wasmPath) {
    // Load WASM module
    const response = await fetch(wasmPath);
    const wasmBinary = await response.arrayBuffer();

    this.module = await createLlamaModule({
      wasmBinary: wasmBinary
    });

    console.log('WASM module loaded');
    return true;
  }

  async loadModel(modelUrl) {
    // Fetch model file
    const response = await fetch(modelUrl);
    const arrayBuffer = await response.arrayBuffer();
    const data = new Uint8Array(arrayBuffer);

    // Allocate memory in WASM
    const modelSize = data.length;
    const modelPtr = this.module._malloc(modelSize);

    // Copy model to WASM memory
    this.module.HEAPU8.set(data, modelPtr);

    // Initialize model
    this.context = this.module.ccall(
      'llama_init',
      'number',
      ['number', 'number'],
      [modelPtr, modelSize]
    );

    if (!this.context) {
      throw new Error('Failed to load model');
    }

    return true;
  }

  generate(prompt, maxTokens = 512, onToken) {
    if (!this.context) {
      throw new Error('Model not loaded');
    }

    // Convert prompt to C string
    const promptPtr = this.module.allocateUTF8(prompt);

    // Create callback
    const callback = this.module.addFunction((tokenPtr) => {
      const token = this.module.UTF8ToString(tokenPtr);
      if (onToken) onToken(token);
    }, 'vi');

    // Call generate function
    this.module.ccall(
      'llama_generate',
      'void',
      ['number', 'number', 'number', 'number'],
      [this.context, promptPtr, maxTokens, callback]
    );

    // Cleanup
    this.module._free(promptPtr);
    this.module.removeFunction(callback);
  }

  free() {
    if (this.context) {
      this.module.ccall('llama_free', 'void', ['number'], [this.context]);
      this.context = null;
    }
  }
}

export default LlamaWasm;
```

### TypeScript Wrapper

```typescript
// llama.ts
export interface ModelConfig {
  wasmPath: string;
  modelPath: string;
  nCtx?: number;
  nThreads?: number;
}

export interface GenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export class LlamaCpp {
  private wasm: LlamaWasm;
  private loaded: boolean = false;

  constructor() {
    this.wasm = new LlamaWasm();
  }

  async load(config: ModelConfig): Promise<void> {
    // Initialize WASM
    await this.wasm.initialize(config.wasmPath);

    // Load model
    await this.wasm.loadModel(config.modelPath);

    this.loaded = true;
  }

  async* generate(
    prompt: string,
    config: GenerationConfig = {}
  ): AsyncGenerator<string> {
    if (!this.loaded) {
      throw new Error('Model not loaded');
    }

    const tokens: string[] = [];

    return new Promise((resolve, reject) => {
      try {
        this.wasm.generate(
          prompt,
          config.maxTokens || 512,
          (token: string) => {
            tokens.push(token);
          }
        );

        // Return generator
        resolve(
          (async function* () {
            for (const token of tokens) {
              yield token;
            }
          })()
        );
      } catch (error) {
        reject(error);
      }
    });
  }

  dispose(): void {
    this.wasm.free();
    this.loaded = false;
  }
}
```

---

## 3. Web Worker Integration

### Worker for Background Inference

```javascript
// llama-worker.js
importScripts('llama.js');

let llama = null;

self.addEventListener('message', async (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'init':
      try {
        llama = new LlamaWasm();
        await llama.initialize(data.wasmPath);
        await llama.loadModel(data.modelPath);

        self.postMessage({ type: 'ready' });
      } catch (error) {
        self.postMessage({
          type: 'error',
          error: error.message
        });
      }
      break;

    case 'generate':
      try {
        llama.generate(
          data.prompt,
          data.maxTokens,
          (token) => {
            self.postMessage({
              type: 'token',
              token: token
            });
          }
        );

        self.postMessage({ type: 'done' });
      } catch (error) {
        self.postMessage({
          type: 'error',
          error: error.message
        });
      }
      break;

    case 'terminate':
      llama.free();
      self.close();
      break;
  }
});
```

### Main Thread Interface

```javascript
// llama-client.js
export class LlamaClient {
  constructor() {
    this.worker = new Worker('llama-worker.js');
    this.ready = false;
    this.callbacks = {};

    this.worker.addEventListener('message', (event) => {
      this.handleMessage(event.data);
    });
  }

  async initialize(wasmPath, modelPath) {
    return new Promise((resolve, reject) => {
      this.callbacks['ready'] = resolve;
      this.callbacks['error'] = reject;

      this.worker.postMessage({
        type: 'init',
        data: { wasmPath, modelPath }
      });
    });
  }

  async* generate(prompt, maxTokens = 512) {
    const tokens = [];

    return new Promise((resolve, reject) => {
      this.callbacks['token'] = (token) => {
        tokens.push(token);
      };

      this.callbacks['done'] = () => {
        resolve(
          (async function* () {
            for (const token of tokens) {
              yield token;
            }
          })()
        );
      };

      this.callbacks['error'] = reject;

      this.worker.postMessage({
        type: 'generate',
        data: { prompt, maxTokens }
      });
    });
  }

  handleMessage(message) {
    const { type } = message;

    if (this.callbacks[type]) {
      if (type === 'token') {
        this.callbacks[type](message.token);
      } else if (type === 'error') {
        this.callbacks[type](new Error(message.error));
      } else {
        this.callbacks[type]();
      }
    }
  }

  terminate() {
    this.worker.postMessage({ type: 'terminate' });
  }
}
```

---

## 4. Browser Chat Application

### HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLaMA Web Chat</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f5f5f5;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .container {
      width: 90%;
      max-width: 800px;
      height: 90vh;
      background: white;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
    }

    .header {
      padding: 20px;
      border-bottom: 1px solid #eee;
    }

    .status {
      font-size: 14px;
      color: #666;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    }

    .message {
      margin-bottom: 15px;
      display: flex;
    }

    .message.user {
      justify-content: flex-end;
    }

    .message-content {
      max-width: 70%;
      padding: 10px 15px;
      border-radius: 10px;
      line-height: 1.4;
    }

    .message.user .message-content {
      background: #007bff;
      color: white;
    }

    .message.assistant .message-content {
      background: #f1f1f1;
      color: #333;
    }

    .input-area {
      border-top: 1px solid #eee;
      padding: 20px;
      display: flex;
      gap: 10px;
    }

    input {
      flex: 1;
      padding: 10px 15px;
      border: 1px solid #ddd;
      border-radius: 5px;
      font-size: 14px;
    }

    button {
      padding: 10px 20px;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 14px;
    }

    button:hover:not(:disabled) {
      background: #0056b3;
    }

    button:disabled {
      background: #ccc;
      cursor: not-allowed;
    }

    .loading {
      text-align: center;
      padding: 40px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>LLaMA Web Chat</h1>
      <div class="status" id="status">Initializing...</div>
    </div>

    <div class="messages" id="messages"></div>

    <div class="input-area">
      <input
        type="text"
        id="input"
        placeholder="Type a message..."
        disabled
      />
      <button id="send" disabled>Send</button>
    </div>
  </div>

  <script type="module" src="app.js"></script>
</body>
</html>
```

### Application Logic

```javascript
// app.js
import { LlamaClient } from './llama-client.js';

class ChatApp {
  constructor() {
    this.llama = new LlamaClient();
    this.messages = [];
    this.isGenerating = false;

    this.setupUI();
    this.initialize();
  }

  setupUI() {
    this.statusEl = document.getElementById('status');
    this.messagesEl = document.getElementById('messages');
    this.inputEl = document.getElementById('input');
    this.sendBtn = document.getElementById('send');

    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.inputEl.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.sendMessage();
    });
  }

  async initialize() {
    try {
      this.statusEl.textContent = 'Loading model...';

      await this.llama.initialize(
        './llama.wasm',
        './models/model.gguf'
      );

      this.statusEl.textContent = 'Ready';
      this.inputEl.disabled = false;
      this.sendBtn.disabled = false;
      this.inputEl.focus();
    } catch (error) {
      this.statusEl.textContent = `Error: ${error.message}`;
    }
  }

  async sendMessage() {
    const text = this.inputEl.value.trim();
    if (!text || this.isGenerating) return;

    this.isGenerating = true;
    this.inputEl.value = '';
    this.sendBtn.disabled = true;

    // Add user message
    this.addMessage(text, true);

    try {
      // Generate response
      let response = '';
      const messageEl = this.addMessage('', false);

      const stream = await this.llama.generate(text);
      for await (const token of stream) {
        response += token;
        messageEl.textContent = response;
        this.scrollToBottom();
      }
    } catch (error) {
      this.addMessage(`Error: ${error.message}`, false);
    }

    this.isGenerating = false;
    this.sendBtn.disabled = false;
    this.inputEl.focus();
  }

  addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    messageDiv.appendChild(contentDiv);
    this.messagesEl.appendChild(messageDiv);

    this.scrollToBottom();

    return contentDiv;
  }

  scrollToBottom() {
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }
}

// Initialize app
new ChatApp();
```

---

## 5. Performance Optimization

### Progressive Model Loading

```javascript
class ProgressiveLoader {
  async loadModel(url, onProgress) {
    const response = await fetch(url);
    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength, 10);

    let loaded = 0;
    const reader = response.body.getReader();
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (onProgress) {
        onProgress({
          loaded,
          total,
          percentage: (loaded / total) * 100
        });
      }
    }

    // Combine chunks
    const buffer = new Uint8Array(loaded);
    let position = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, position);
      position += chunk.length;
    }

    return buffer;
  }
}

// Usage
const loader = new ProgressiveLoader();
const model = await loader.loadModel(
  './models/model.gguf',
  (progress) => {
    console.log(`Loading: ${progress.percentage.toFixed(1)}%`);
  }
);
```

### IndexedDB Caching

```javascript
class ModelCache {
  constructor(dbName = 'llama-models') {
    this.dbName = dbName;
    this.db = null;
  }

  async initialize() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'name' });
        }
      };
    });
  }

  async getModel(name) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['models'], 'readonly');
      const store = transaction.objectStore('models');
      const request = store.get(name);

      request.onsuccess = () => resolve(request.result?.data);
      request.onerror = () => reject(request.error);
    });
  }

  async setModel(name, data) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['models'], 'readwrite');
      const store = transaction.objectStore('models');
      const request = store.put({ name, data });

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async loadWithCache(name, url, onProgress) {
    // Check cache first
    const cached = await this.getModel(name);
    if (cached) {
      console.log('Model loaded from cache');
      return cached;
    }

    // Download and cache
    const loader = new ProgressiveLoader();
    const data = await loader.loadModel(url, onProgress);

    await this.setModel(name, data);

    return data;
  }
}
```

---

## 6. React Integration

### React Hook

```typescript
// useLlama.ts
import { useState, useEffect, useCallback } from 'react';
import { LlamaClient } from './llama-client';

interface UseLlamaOptions {
  wasmPath: string;
  modelPath: string;
}

export function useLlama(options: UseLlamaOptions) {
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [llama] = useState(() => new LlamaClient());

  useEffect(() => {
    const init = async () => {
      try {
        await llama.initialize(options.wasmPath, options.modelPath);
        setIsReady(true);
      } catch (err) {
        setError(err.message);
      }
    };

    init();

    return () => {
      llama.terminate();
    };
  }, [options.wasmPath, options.modelPath]);

  const generate = useCallback(
    async function* (prompt: string, maxTokens = 512) {
      if (!isReady) {
        throw new Error('Model not ready');
      }

      setIsGenerating(true);
      try {
        const stream = await llama.generate(prompt, maxTokens);
        for await (const token of stream) {
          yield token;
        }
      } finally {
        setIsGenerating(false);
      }
    },
    [isReady, llama]
  );

  return {
    isReady,
    isGenerating,
    error,
    generate
  };
}
```

### React Component

```typescript
// Chat.tsx
import React, { useState } from 'react';
import { useLlama } from './useLlama';

export function Chat() {
  const { isReady, isGenerating, generate } = useLlama({
    wasmPath: '/llama.wasm',
    modelPath: '/models/model.gguf'
  });

  const [messages, setMessages] = useState<Array<{text: string, isUser: boolean}>>([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);

    let response = '';
    setMessages(prev => [...prev, { text: '', isUser: false }]);

    for await (const token of generate(userMessage)) {
      response += token;
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1].text = response;
        return newMessages;
      });
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.isUser ? 'user' : 'assistant'}`}>
            {msg.text}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          disabled={!isReady || isGenerating}
          placeholder="Type a message..."
        />
        <button
          onClick={sendMessage}
          disabled={!isReady || isGenerating || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
```

---

## Summary

In this lesson, you learned:
- ✅ Compiling llama.cpp to WebAssembly
- ✅ JavaScript/TypeScript bindings
- ✅ Web Worker for background processing
- ✅ Browser-based chat applications
- ✅ Performance optimization and caching
- ✅ React integration

## Next Steps

- **Lab 8.5**: Build a web-based chat app
- **Project**: Browser extension with LLM
- Review all Module 8 content

## Additional Resources

- [Emscripten Documentation](https://emscripten.org/docs/)
- [WebAssembly Guide](https://webassembly.org/getting-started/developers-guide/)
- [Web Workers MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)

---

**Module**: 08 - Integration & Applications
**Lesson**: 06 - Web Integration
**Version**: 1.0
**Last Updated**: 2025-11-18
