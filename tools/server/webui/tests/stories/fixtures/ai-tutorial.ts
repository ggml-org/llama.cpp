// AI Assistant Tutorial Response
export const AI_TUTORIAL_MD = String.raw`
# Building a Modern Chat Application with SvelteKit

I'll help you create a **production-ready chat application** using SvelteKit, TypeScript, and WebSockets. This implementation includes real-time messaging, user authentication, and message persistence.

## 🚀 Quick Start

First, let's set up the project:

${'```'}bash
npm create svelte@latest chat-app
cd chat-app
npm install
npm install socket.io socket.io-client
npm install @prisma/client prisma
npm run dev
${'```'}

## 📁 Project Structure

${'```'}
chat-app/
├── src/
│   ├── routes/
│   │   ├── +layout.svelte
│   │   ├── +page.svelte
│   │   └── api/
│   │       └── socket/+server.ts
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ChatMessage.svelte
│   │   │   └── ChatInput.svelte
│   │   └── stores/
│   │       └── chat.ts
│   └── app.html
├── prisma/
│   └── schema.prisma
└── package.json
${'```'}

## 💻 Implementation

### WebSocket Server

${'```'}typescript
// src/lib/server/socket.ts
import { Server } from 'socket.io';
import type { ViteDevServer } from 'vite';

export function initializeSocketIO(server: ViteDevServer) {
    const io = new Server(server.httpServer || server, {
        cors: {
            origin: process.env.ORIGIN || 'http://localhost:5173',
            credentials: true
        }
    });

    io.on('connection', (socket) => {
        console.log('User connected:', socket.id);
        
        socket.on('message', async (data) => {
            // Broadcast to all clients
            io.emit('new-message', {
                id: crypto.randomUUID(),
                userId: socket.id,
                content: data.content,
                timestamp: new Date().toISOString()
            });
        });

        socket.on('disconnect', () => {
            console.log('User disconnected:', socket.id);
        });
    });

    return io;
}
${'```'}

### Client Store

${'```'}typescript
// src/lib/stores/chat.ts
import { writable } from 'svelte/store';
import io from 'socket.io-client';

export interface Message {
    id: string;
    userId: string;
    content: string;
    timestamp: string;
}

function createChatStore() {
    const { subscribe, update } = writable<Message[]>([]);
    let socket: ReturnType<typeof io>;
    
    return {
        subscribe,
        connect: () => {
            socket = io('http://localhost:5173');
            
            socket.on('new-message', (message: Message) => {
                update(messages => [...messages, message]);
            });
        },
        sendMessage: (content: string) => {
            if (socket && content.trim()) {
                socket.emit('message', { content });
            }
        }
    };
}

export const chatStore = createChatStore();
${'```'}

## 🎯 Key Features

✅ **Real-time messaging** with WebSockets  
✅ **Message persistence** using Prisma + PostgreSQL  
✅ **Type-safe** with TypeScript  
✅ **Responsive UI** for all devices  
✅ **Auto-reconnection** on connection loss  

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Message Latency** | < 50ms |
| **Concurrent Users** | 10,000+ |
| **Messages/Second** | 5,000+ |
| **Uptime** | 99.9% |

## 🔧 Configuration

### Environment Variables

${'```'}env
DATABASE_URL="postgresql://user:password@localhost:5432/chat"
JWT_SECRET="your-secret-key"
REDIS_URL="redis://localhost:6379"
${'```'}

## 🚢 Deployment

Deploy to production using Docker:

${'```'}dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["node", "build"]
${'```'}

---

*Need help? Check the [documentation](https://kit.svelte.dev) or [open an issue](https://github.com/sveltejs/kit/issues)*
`;
