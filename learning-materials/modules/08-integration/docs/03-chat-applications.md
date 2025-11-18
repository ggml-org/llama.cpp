# Building Chat Applications

**Module 8, Lesson 3**
**Estimated Time**: 3-4 hours
**Difficulty**: Intermediate

## Overview

Learn to build production-ready chat applications using llama.cpp, from simple CLI chatbots to full-featured web interfaces with streaming, conversation management, and multi-user support.

## Learning Objectives

- Build CLI and web-based chat interfaces
- Implement conversation history and context management
- Handle streaming responses
- Create multi-user chat systems
- Optimize for real-time performance
- Deploy chat applications to production

## Prerequisites

- Module 8, Lesson 1: Python Bindings
- Web development basics (for web chat)
- Understanding of async programming

---

## 1. Chat Application Architecture

### Core Components

```
┌────────────────────────────────────────┐
│         Chat Application               │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  User Interface                  │ │
│  │  - CLI / Web / Mobile            │ │
│  └────────────┬─────────────────────┘ │
│               │                        │
│               ▼                        │
│  ┌──────────────────────────────────┐ │
│  │  Conversation Manager            │ │
│  │  - History tracking              │ │
│  │  - Context windowing             │ │
│  └────────────┬─────────────────────┘ │
│               │                        │
│               ▼                        │
│  ┌──────────────────────────────────┐ │
│  │  Prompt Builder                  │ │
│  │  - Template formatting           │ │
│  │  - System prompts                │ │
│  └────────────┬─────────────────────┘ │
│               │                        │
│               ▼                        │
│  ┌──────────────────────────────────┐ │
│  │  LLM Engine (llama.cpp)          │ │
│  │  - Text generation               │ │
│  │  - Streaming support             │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
```

---

## 2. CLI Chat Application

### Basic CLI Chatbot

```python
#!/usr/bin/env python3
"""Simple CLI chatbot using llama.cpp."""

from llama_cpp import Llama
from typing import List, Dict

class CLIChatbot:
    """Command-line chatbot."""

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=35,
            chat_format="llama-2",
            verbose=False
        )
        self.history: List[Dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        """Generate response to user message."""
        # Add user message to history
        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Generate response
        response = self.llm.create_chat_completion(
            messages=self.history,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95
        )

        assistant_message = response['choices'][0]['message']['content']

        # Add assistant response to history
        self.history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def run(self):
        """Run interactive chat loop."""
        print("Chatbot initialized. Type 'quit' to exit.")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Generate and print response
                response = self.chat(user_input)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

if __name__ == '__main__':
    bot = CLIChatbot(model_path="./models/llama-2-7b-chat.Q4_K_M.gguf")
    bot.run()
```

### Streaming CLI Chatbot

```python
#!/usr/bin/env python3
"""CLI chatbot with streaming responses."""

from llama_cpp import Llama
import sys

class StreamingCLIChatbot:
    """Chatbot with streaming output."""

    def __init__(self, model_path: str, system_prompt: str = None):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=35,
            chat_format="llama-2"
        )
        self.history = []

        if system_prompt:
            self.history.append({
                "role": "system",
                "content": system_prompt
            })

    def chat_stream(self, user_message: str):
        """Generate streaming response."""
        self.history.append({"role": "user", "content": user_message})

        # Generate streaming response
        stream = self.llm.create_chat_completion(
            messages=self.history,
            max_tokens=512,
            temperature=0.7,
            stream=True
        )

        # Collect full response
        full_response = ""

        print("\nAssistant: ", end='', flush=True)
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                print(content, end='', flush=True)
                full_response += content

        print()  # Newline after response

        # Add to history
        self.history.append({
            "role": "assistant",
            "content": full_response
        })

    def run(self):
        """Run chat loop."""
        print("Streaming Chatbot (type 'quit' to exit)")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    break

                if user_input:
                    self.chat_stream(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

if __name__ == '__main__':
    bot = StreamingCLIChatbot(
        model_path="./models/model.gguf",
        system_prompt="You are a helpful, friendly assistant."
    )
    bot.run()
```

---

## 3. Web Chat Application

### Flask Web Chat

```python
from flask import Flask, render_template, request, jsonify, Response
from llama_cpp import Llama
import json
from datetime import datetime

app = Flask(__name__)

# Initialize model
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    chat_format="llama-2"
)

# Store conversations (in production, use Redis/database)
conversations = {}

@app.route('/')
def index():
    """Serve chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')

    # Get or create conversation history
    if session_id not in conversations:
        conversations[session_id] = []

    # Add user message
    conversations[session_id].append({
        "role": "user",
        "content": message
    })

    # Generate response
    response = llm.create_chat_completion(
        messages=conversations[session_id],
        max_tokens=512,
        temperature=0.7
    )

    assistant_message = response['choices'][0]['message']['content']

    # Add to history
    conversations[session_id].append({
        "role": "assistant",
        "content": assistant_message
    })

    return jsonify({
        'response': assistant_message,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint."""
    data = request.json
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')

    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": "user",
        "content": message
    })

    def generate():
        stream = llm.create_chat_completion(
            messages=conversations[session_id],
            max_tokens=512,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"

        # Add complete response to history
        conversations[session_id].append({
            "role": "assistant",
            "content": full_response
        })

        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### HTML Chat Interface

```html
<!-- templates/chat.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-container {
            width: 800px;
            height: 600px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 20px;
            border-bottom: 1px solid #eee;
        }

        .chat-header h1 {
            font-size: 24px;
            color: #333;
        }

        .chat-messages {
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

        .chat-input {
            border-top: 1px solid #eee;
            padding: 20px;
            display: flex;
            gap: 10px;
        }

        .chat-input input {
            flex: 1;
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        .chat-input button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .chat-input button:hover {
            background: #0056b3;
        }

        .chat-input button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>LLaMA Chat</h1>
        </div>

        <div class="chat-messages" id="messages"></div>

        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Type your message..." />
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const sessionId = 'session-' + Date.now();

        function addMessage(content, role) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;

            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            return contentDiv;
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            messageInput.value = '';
            sendButton.disabled = true;

            // Create assistant message div for streaming
            const assistantContent = addMessage('', 'assistant');

            // Stream response
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        if (data.content) {
                            assistantContent.textContent += data.content;
                            messagesDiv.scrollTop = messagesDiv.scrollHeight;
                        }
                    }
                }
            }

            sendButton.disabled = false;
            messageInput.focus();
        }

        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

---

## 4. FastAPI Chat Application

### Async Chat Server

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
from typing import List, Dict
import json
import asyncio
from datetime import datetime

app = FastAPI()

# Initialize model
llm = Llama(
    model_path="./models/model.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    chat_format="llama-2"
)

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversations: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.conversations:
            self.conversations[client_id] = []

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get('message', '')

            # Add to history
            manager.conversations[client_id].append({
                "role": "user",
                "content": user_message
            })

            # Generate response in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: llm.create_chat_completion(
                    messages=manager.conversations[client_id],
                    max_tokens=512,
                    stream=True
                )
            )

            # Stream response
            full_response = ""
            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content = delta['content']
                    full_response += content
                    await manager.send_message(
                        json.dumps({
                            'type': 'token',
                            'content': content
                        }),
                        websocket
                    )

            # Add complete response to history
            manager.conversations[client_id].append({
                "role": "assistant",
                "content": full_response
            })

            # Send completion signal
            await manager.send_message(
                json.dumps({'type': 'done'}),
                websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>WebSocket Chat</title>
        </head>
        <body>
            <h1>WebSocket Chat</h1>
            <div id="messages"></div>
            <input type="text" id="messageInput" />
            <button onclick="sendMessage()">Send</button>

            <script>
                const ws = new WebSocket("ws://localhost:8000/ws/client123");
                const messagesDiv = document.getElementById('messages');
                let currentMessage = null;

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);

                    if (data.type === 'token') {
                        if (!currentMessage) {
                            currentMessage = document.createElement('div');
                            messagesDiv.appendChild(currentMessage);
                        }
                        currentMessage.textContent += data.content;
                    } else if (data.type === 'done') {
                        currentMessage = null;
                    }
                };

                function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value;

                    // Display user message
                    const userDiv = document.createElement('div');
                    userDiv.textContent = 'You: ' + message;
                    messagesDiv.appendChild(userDiv);

                    // Send to server
                    ws.send(JSON.stringify({message: message}));
                    input.value = '';
                }
            </script>
        </body>
    </html>
    """)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 5. Conversation Management

### Context Window Management

```python
from typing import List, Dict

class ConversationManager:
    """Manage conversation history with context window limits."""

    def __init__(
        self,
        max_tokens: int = 2048,
        system_prompt: str = None
    ):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

    def add_message(self, role: str, content: str):
        """Add message to history."""
        self.messages.append({
            "role": role,
            "content": content
        })
        self._trim_to_limit()

    def _trim_to_limit(self):
        """Trim messages to fit context window."""
        # Simple token estimation (4 chars ≈ 1 token)
        total_tokens = sum(len(msg['content']) // 4 for msg in self.messages)

        # Keep removing oldest messages (except system prompt)
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            # Remove oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg['role'] != 'system':
                    self.messages.pop(i)
                    total_tokens -= len(msg['content']) // 4
                    break

    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history."""
        return self.messages.copy()

    def clear(self):
        """Clear history (keep system prompt)."""
        system_msgs = [msg for msg in self.messages if msg['role'] == 'system']
        self.messages = system_msgs
```

### Multi-User Chat System

```python
from typing import Dict
from datetime import datetime
import uuid

class MultiUserChatSystem:
    """Manage multiple concurrent chat sessions."""

    def __init__(self, llm: Llama):
        self.llm = llm
        self.sessions: Dict[str, ConversationManager] = {}

    def create_session(self, system_prompt: str = None) -> str:
        """Create new chat session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ConversationManager(
            max_tokens=2048,
            system_prompt=system_prompt
        )
        return session_id

    def delete_session(self, session_id: str):
        """Delete chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def chat(
        self,
        session_id: str,
        message: str,
        stream: bool = False
    ):
        """Send message in session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        manager = self.sessions[session_id]
        manager.add_message("user", message)

        response = self.llm.create_chat_completion(
            messages=manager.get_messages(),
            max_tokens=512,
            stream=stream
        )

        if stream:
            def generator():
                full_response = ""
                for chunk in response:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        content = delta['content']
                        full_response += content
                        yield content

                # Add to history after completion
                manager.add_message("assistant", full_response)

            return generator()
        else:
            assistant_message = response['choices'][0]['message']['content']
            manager.add_message("assistant", assistant_message)
            return assistant_message
```

---

## Summary

In this lesson, you learned:
- ✅ Building CLI chat interfaces with streaming
- ✅ Creating web chat applications with Flask/FastAPI
- ✅ Implementing WebSocket real-time chat
- ✅ Managing conversation history and context
- ✅ Multi-user chat systems

## Next Steps

- **Lesson 4**: Function Calling - Add tool use to chat applications
- **Lab 8.3**: Build a production chat application
- **Project**: Multi-platform chat app

## Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

---

**Module**: 08 - Integration & Applications
**Lesson**: 03 - Building Chat Applications
**Version**: 1.0
**Last Updated**: 2025-11-18
