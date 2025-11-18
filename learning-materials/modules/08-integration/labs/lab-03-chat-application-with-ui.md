# Lab 8.3: Chat Application with UI

**Estimated Time**: 3 hours
**Difficulty**: Intermediate
**Prerequisites**: Lessons 8.1, 8.3

## Objective

Build a full-featured web chat application with Flask/FastAPI backend and modern web frontend, supporting streaming, conversation history, and multi-user sessions.

## Part 1: Backend API (60 min)

### FastAPI Implementation

```python
# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
import json
import uuid

app = FastAPI()

# TODO: Initialize model
llm = Llama(model_path="./models/model.gguf", n_ctx=4096)

# TODO: Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.conversations = {}
    # Complete implementation

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # TODO: Implement WebSocket chat
    pass

@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    # TODO: Implement REST chat endpoint
    pass
```

**Exercise**: Complete the WebSocket and REST endpoints.

## Part 2: Frontend (60 min)

### HTML/CSS/JavaScript Chat Interface

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>LLaMA Chat</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input id="message-input" type="text" />
            <button id="send-btn">Send</button>
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
```

```css
/* static/style.css */
/* TODO: Style the chat interface */
#chat-container {
    max-width: 800px;
    margin: 0 auto;
    height: 100vh;
}

/* Add your styles */
```

```javascript
// static/app.js
class ChatClient {
    constructor() {
        this.ws = null;
        this.sessionId = this.generateSessionId();
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(`ws://localhost:8000/ws/${this.sessionId}`);
        // TODO: Implement WebSocket handlers
    }

    sendMessage(message) {
        // TODO: Implement message sending
    }
}

// Initialize
const chat = new ChatClient();
```

**Exercise**: Complete the chat interface with streaming support.

## Part 3: Features (60 min)

### Conversation Management

```python
# conversation.py
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.conversations = {}
        self.max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        # TODO: Add message to history
        pass

    def get_conversation(self, session_id: str) -> List[Dict]:
        # TODO: Return conversation history
        pass

    def clear_conversation(self, session_id: str):
        # TODO: Clear history
        pass

    def trim_history(self, session_id: str, max_tokens: int):
        # TODO: Trim to fit context window
        pass
```

### Add These Features:
1. Conversation persistence (save/load)
2. Multiple conversation threads
3. Message editing and regeneration
4. Export conversation to markdown

## Part 4: Testing (30 min)

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_chat_endpoint():
    response = client.post("/api/chat", json={
        "message": "Hello",
        "session_id": "test-123"
    })
    assert response.status_code == 200
    assert "response" in response.json()

# TODO: Add WebSocket tests
```

## Challenges

1. **Multi-User Support**: Handle concurrent users
2. **Authentication**: Add user login
3. **Model Selection**: Allow users to choose models
4. **Voice Input**: Add speech-to-text
5. **Code Highlighting**: Format code blocks in responses

## Success Criteria

- [X] Backend API functional
- [X] Frontend UI responsive
- [X] WebSocket streaming works
- [X] Conversation history managed
- [X] Tests passing
- [X] Multi-user support

## Submission

Submit:
1. Complete application code
2. Screenshots of UI
3. API documentation
4. Demo video

---

**Lab**: 8.3 - Chat Application with UI
**Module**: 08 - Integration & Applications
**Version**: 1.0
