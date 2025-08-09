<script module lang="ts">
    import { defineMeta } from '@storybook/addon-svelte-csf';
    import { MarkdownContent } from '$lib/components/app';

    const { Story } = defineMeta({
        title: 'Components/MarkdownContent',
        component: MarkdownContent,
        parameters: {
            layout: 'centered'
        }
    });

    // AI Assistant Tutorial Response
    const AI_TUTORIAL_MD = String.raw`
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

    // API Documentation
    const API_DOCS_MD = String.raw`
# REST API Documentation

## 🔐 Authentication

All API requests require authentication using **Bearer tokens**. Include your API key in the Authorization header:

${'```'}http
GET /api/v1/users
Host: api.example.com
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
${'```'}

## 📍 Endpoints

### Users API

#### **GET** /api/v1/users

Retrieve a paginated list of users.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| limit | integer | 20 | Items per page |
| sort | string | "created_at" | Sort field |
| order | string | "desc" | Sort order |

**Response:** 200 OK

${'```'}json
{
  "data": [
    {
      "id": "usr_1234567890",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "admin",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "pages": 8
  }
}
${'```'}

#### **POST** /api/v1/users

Create a new user account.

**Request Body:**

${'```'}json
{
  "email": "newuser@example.com",
  "password": "SecurePassword123!",
  "name": "Jane Smith",
  "role": "user"
}
${'```'}

**Response:** 201 Created

${'```'}json
{
  "id": "usr_9876543210",
  "email": "newuser@example.com",
  "name": "Jane Smith",
  "role": "user",
  "created_at": "2024-01-21T09:15:00Z"
}
${'```'}

### Error Responses

The API returns errors in a consistent format:

${'```'}json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Email format is invalid"
      }
    ]
  }
}
${'```'}

### Rate Limiting

| Tier | Requests/Hour | Burst |
|------|--------------|-------|
| **Free** | 1,000 | 100 |
| **Pro** | 10,000 | 500 |
| **Enterprise** | Unlimited | - |

**Headers:**
- X-RateLimit-Limit
- X-RateLimit-Remaining  
- X-RateLimit-Reset

### Webhooks

Configure webhooks to receive real-time events:

${'```'}javascript
// Webhook payload
{
  "event": "user.created",
  "timestamp": "2024-01-21T09:15:00Z",
  "data": {
    "id": "usr_9876543210",
    "email": "newuser@example.com"
  },
  "signature": "sha256=abcd1234..."
}
${'```'}

### SDK Examples

**JavaScript/TypeScript:**

${'```'}typescript
import { ApiClient } from '@example/api-sdk';

const client = new ApiClient({
  apiKey: process.env.API_KEY
});

const users = await client.users.list({
  page: 1,
  limit: 20
});
${'```'}

**Python:**

${'```'}python
from example_api import Client

client = Client(api_key=os.environ['API_KEY'])
users = client.users.list(page=1, limit=20)
${'```'}

---

📚 [Full API Reference](https://api.example.com/docs) | 💬 [Support](https://support.example.com)
`;

    // Technical Blog Post
    const BLOG_POST_MD = String.raw`
# Understanding Rust's Ownership System

*Published on January 21, 2024 • 8 min read*

## Introduction

Rust's **ownership system** is its most distinctive feature, enabling memory safety without garbage collection. Let's explore how ownership, borrowing, and lifetimes work together.

## The Three Rules

1. Each value has a **single owner**
2. There can be **only one owner** at a time
3. When the owner goes out of scope, the value is **dropped**

## Code Examples

### Basic Ownership

${'```'}rust
fn main() {
    let s1 = String::from("hello");  // s1 owns the String
    let s2 = s1;                     // Ownership moves to s2
    // println!("{}", s1);           // ❌ ERROR: s1 no longer valid
    println!("{}", s2);              // ✅ OK: s2 owns the String
}
${'```'}

### Borrowing

${'```'}rust
fn calculate_length(s: &String) -> usize {
    s.len()  // s is a reference, doesn't own the String
}

fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // Borrow s1
    println!("Length of '{}' is {}", s1, len);  // s1 still valid
}
${'```'}

### Mutable References

${'```'}rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &mut s;
    r1.push_str(", world");
    println!("{}", r1);
    
    // let r2 = &mut s;  // ❌ ERROR: cannot borrow twice
}
${'```'}

## Real-World Example: Smart Cache

${'```'}rust
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

struct Cache<T> {
    storage: RefCell<HashMap<String, Rc<T>>>,
}

impl<T> Cache<T> {
    fn new() -> Self {
        Cache {
            storage: RefCell::new(HashMap::new()),
        }
    }
    
    fn get(&self, key: &str) -> Option<Rc<T>> {
        self.storage.borrow().get(key).cloned()
    }
    
    fn set(&self, key: String, value: T) {
        self.storage.borrow_mut().insert(key, Rc::new(value));
    }
}
${'```'}

## Performance Comparison

| Operation | Rust | C++ | Go | Java |
|-----------|------|-----|-----|------|
| **String concat** | 12ns | 15ns | 22ns | 28ns |
| **Vector push** | 5ns | 6ns | 8ns | 10ns |
| **HashMap insert** | 35ns | 40ns | 45ns | 52ns |

## Common Pitfalls

### ❌ Dangling References

${'```'}rust
fn dangle() -> &String {
    let s = String::from("hello");
    &s  // ERROR: s goes out of scope
}
${'```'}

### ✅ Solution

${'```'}rust
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // Ownership is moved out
}
${'```'}

## Benefits

- ✅ **No null pointer dereferences**
- ✅ **No data races**
- ✅ **No use-after-free**
- ✅ **No memory leaks**

## Conclusion

Rust's ownership system eliminates entire classes of bugs at compile time. While it has a learning curve, the benefits in safety and performance are worth it.

## Further Reading

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rustonomicon](https://doc.rust-lang.org/nomicon/)

---

*Follow me on [Twitter](https://twitter.com/rustdev) • [GitHub](https://github.com/rustdev)*
`;

    // Data Analysis Report
    const DATA_ANALYSIS_MD = String.raw`
# Q4 2023 Performance Report

## Executive Summary

Our analysis reveals **strong growth** across all key metrics, with revenue up **42% YoY** and user engagement reaching all-time highs.

## 📊 Key Metrics

### Revenue Performance

| Quarter | Revenue | Growth | Target | Achievement |
|---------|---------|--------|--------|-------------|
| Q1 2023 | $2.4M | +15% | $2.2M | 109% |
| Q2 2023 | $2.8M | +22% | $2.5M | 112% |
| Q3 2023 | $3.2M | +31% | $3.0M | 107% |
| **Q4 2023** | **$4.1M** | **+42%** | **$3.5M** | **117%** |

### User Metrics

${'```'}
Daily Active Users (DAU):   1.2M (+65% YoY)
Monthly Active Users (MAU): 4.5M (+48% YoY)
User Retention (Day 30):    68% (+12pp YoY)
Average Session Duration:   24min (+35% YoY)
${'```'}

## 🎯 Product Performance

### Feature Adoption Rates

1. **AI Assistant**: 78% of users (↑ from 45%)
2. **Collaboration Tools**: 62% of users (↑ from 38%)
3. **Analytics Dashboard**: 54% of users (↑ from 31%)
4. **Mobile App**: 41% of users (↑ from 22%)

### Customer Satisfaction

> "The platform has transformed how our team works. We've seen a **3x improvement** in productivity."
> — Sarah Chen, CTO at TechCorp

**NPS Score:** 72 (Excellent)

## 📈 Growth Drivers

### Geographic Distribution

${'```'}
North America:  45% ($1.85M)
Europe:         28% ($1.15M)
Asia Pacific:   20% ($0.82M)
Other:          7%  ($0.28M)
${'```'}

### Customer Segments

| Segment | Revenue | Customers | ARPU | Churn |
|---------|---------|-----------|------|-------|
| Enterprise | $2.1M | 120 | $17.5K | 2% |
| Mid-Market | $1.4M | 450 | $3.1K | 5% |
| SMB | $0.6M | 2,800 | $214 | 8% |

## 🔮 Projections

### Q1 2024 Forecast

- **Revenue Target**: $4.8M
- **User Growth**: +25% QoQ
- **Market Expansion**: 3 new regions
- **Product Launches**: 2 major features

### Risk Factors

⚠️ **Competition**: New entrants in key markets  
⚠️ **Regulation**: Upcoming data privacy laws  
⚠️ **Technology**: Platform migration challenges  

## 💡 Recommendations

1. **Invest** in AI capabilities (+$2M budget)
2. **Expand** sales team in APAC region
3. **Improve** onboarding to reduce churn
4. **Launch** enterprise security features

## Appendix

### Methodology

Data collected from:
- Internal analytics (Amplitude)
- Customer surveys (n=2,450)
- Financial systems (NetSuite)
- Market research (Gartner)

---

*Report prepared by Data Analytics Team • [View Interactive Dashboard](https://analytics.example.com)*
`;

    // GitHub README
    const README_MD = String.raw`
# 🚀 Awesome Web Framework

[![npm version](https://img.shields.io/npm/v/awesome-framework.svg)](https://www.npmjs.com/package/awesome-framework)
[![Build Status](https://github.com/awesome/framework/workflows/CI/badge.svg)](https://github.com/awesome/framework/actions)
[![Coverage](https://codecov.io/gh/awesome/framework/branch/main/graph/badge.svg)](https://codecov.io/gh/awesome/framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A modern, fast, and flexible web framework for building scalable applications

## ✨ Features

- 🎯 **Type-Safe** - Full TypeScript support out of the box
- ⚡ **Lightning Fast** - Built on Vite for instant HMR
- 📦 **Zero Config** - Works out of the box for most use cases
- 🎨 **Flexible** - Unopinionated with sensible defaults
- 🔧 **Extensible** - Plugin system for custom functionality
- 📱 **Responsive** - Mobile-first approach
- 🌍 **i18n Ready** - Built-in internationalization
- 🔒 **Secure** - Security best practices by default

## 📦 Installation

${'```'}bash
npm install awesome-framework
# or
yarn add awesome-framework
# or
pnpm add awesome-framework
${'```'}

## 🚀 Quick Start

### Create a new project

${'```'}bash
npx create-awesome-app my-app
cd my-app
npm run dev
${'```'}

### Basic Example

${'```'}javascript
import { createApp } from 'awesome-framework';

const app = createApp({
  port: 3000,
  middleware: ['cors', 'helmet', 'compression']
});

app.get('/', (req, res) => {
  res.json({ message: 'Hello World!' });
});

app.listen(() => {
  console.log('Server running on http://localhost:3000');
});
${'```'}

## 📖 Documentation

### Core Concepts

- [Getting Started](https://docs.awesome.dev/getting-started)
- [Configuration](https://docs.awesome.dev/configuration)
- [Routing](https://docs.awesome.dev/routing)
- [Middleware](https://docs.awesome.dev/middleware)
- [Database](https://docs.awesome.dev/database)
- [Authentication](https://docs.awesome.dev/authentication)

### Advanced Topics

- [Performance Optimization](https://docs.awesome.dev/performance)
- [Deployment](https://docs.awesome.dev/deployment)
- [Testing](https://docs.awesome.dev/testing)
- [Security](https://docs.awesome.dev/security)

## 🛠️ Development

### Prerequisites

- Node.js >= 18
- pnpm >= 8

### Setup

${'```'}bash
git clone https://github.com/awesome/framework.git
cd framework
pnpm install
pnpm dev
${'```'}

### Testing

${'```'}bash
pnpm test        # Run unit tests
pnpm test:e2e    # Run end-to-end tests
pnpm test:watch  # Run tests in watch mode
${'```'}

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Contributors

<a href="https://github.com/awesome/framework/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=awesome/framework" />
</a>

## 📊 Benchmarks

| Framework | Requests/sec | Latency (ms) | Memory (MB) |
|-----------|-------------|--------------|-------------|
| **Awesome** | **45,230** | **2.1** | **42** |
| Express | 28,450 | 3.5 | 68 |
| Fastify | 41,200 | 2.3 | 48 |
| Koa | 32,100 | 3.1 | 52 |

*Benchmarks performed on MacBook Pro M2, Node.js 20.x*

## 📝 License

MIT © [Awesome Team](https://github.com/awesome)

## 🙏 Acknowledgments

Special thanks to all our sponsors and contributors who make this project possible.

---

**[Website](https://awesome.dev)** • **[Documentation](https://docs.awesome.dev)** • **[Discord](https://discord.gg/awesome)** • **[Twitter](https://twitter.com/awesomeframework)**
`;

    // Empty state
    const EMPTY_MD = '';
</script>

<Story
    name="Empty"
    args={{ content: EMPTY_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
    name="AI Tutorial"
    args={{ content: AI_TUTORIAL_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
    name="API Documentation"
    args={{ content: API_DOCS_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
    name="Technical Blog"
    args={{ content: BLOG_POST_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
    name="Data Analysis"
    args={{ content: DATA_ANALYSIS_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
    name="GitHub README"
    args={{ content: README_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>
