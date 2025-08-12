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

// API Documentation
export const API_DOCS_MD = String.raw`
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
export const BLOG_POST_MD = String.raw`
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
export const DATA_ANALYSIS_MD = String.raw`
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
|---------|---------|-----------|------|---------|
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
export const README_MD = String.raw`
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

// Math formulas content
export const MATH_FORMULAS_MD = String.raw`
# Mathematical Formulas and Expressions

This document demonstrates various mathematical formulas and expressions rendered in markdown.

## Basic Arithmetic

Simple inline math: $x + y = z$ and $a \cdot b = c$

Block equation:
$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

## Algebra

### Quadratic Formula
The solutions to $ax^2 + bx + c = 0$ are:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

### Binomial Theorem
$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$$

## Calculus

### Derivatives
The derivative of $f(x) = x^n$ is:
$$f'(x) = nx^{n-1}$$

### Integration
$$\int_a^b f(x) \, dx = F(b) - F(a)$$

### Fundamental Theorem of Calculus
$$\frac{d}{dx} \int_a^x f(t) \, dt = f(x)$$

## Linear Algebra

### Matrix Multiplication
If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, then:
$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

### Eigenvalues and Eigenvectors
For a square matrix $A$, if $Av = \lambda v$ for some non-zero vector $v$, then:
- $\lambda$ is an eigenvalue
- $v$ is an eigenvector

## Statistics and Probability

### Normal Distribution
The probability density function is:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

### Bayes' Theorem
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Central Limit Theorem
For large $n$, the sample mean $\bar{X}$ is approximately:
$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

## Trigonometry

### Pythagorean Identity
$$\sin^2\theta + \cos^2\theta = 1$$

### Euler's Formula
$$e^{i\theta} = \cos\theta + i\sin\theta$$

### Taylor Series for Sine
$$\sin x = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots$$

## Complex Analysis

### Complex Numbers
A complex number can be written as:
$$z = a + bi = r e^{i\theta}$$

where $r = |z| = \sqrt{a^2 + b^2}$ and $\theta = \arg(z)$

### Cauchy-Riemann Equations
For a function $f(z) = u(x,y) + iv(x,y)$ to be analytic:
$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

## Differential Equations

### First-order Linear ODE
$$\frac{dy}{dx} + P(x)y = Q(x)$$

Solution: $y = e^{-\int P(x)dx}\left[\int Q(x)e^{\int P(x)dx}dx + C\right]$

### Heat Equation
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

## Number Theory

### Prime Number Theorem
$$\pi(x) \sim \frac{x}{\ln x}$$

where $\pi(x)$ is the number of primes less than or equal to $x$.

### Fermat's Last Theorem
For $n > 2$, there are no positive integers $a$, $b$, and $c$ such that:
$$a^n + b^n = c^n$$

## Set Theory

### De Morgan's Laws
$$\overline{A \cup B} = \overline{A} \cap \overline{B}$$
$$\overline{A \cap B} = \overline{A} \cup \overline{B}$$

## Advanced Topics

### Riemann Zeta Function
$$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}$$

### Maxwell's Equations
$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$
$$\nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$
$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

### Schrödinger Equation
$$i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)$$

## Inline Math Examples

Here are some inline mathematical expressions:

- The golden ratio: $\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$
- Euler's number: $e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n$
- Pi: $\pi = 4 \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$
- Square root of 2: $\sqrt{2} = 1.41421356...$

## Fractions and Radicals

Complex fraction: $\frac{\frac{a}{b} + \frac{c}{d}}{\frac{e}{f} - \frac{g}{h}}$

Nested radicals: $\sqrt{2 + \sqrt{3 + \sqrt{4 + \sqrt{5}}}}$

## Summations and Products

### Geometric Series
$$\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r} \quad \text{for } |r| < 1$$

### Product Notation
$$n! = \prod_{k=1}^{n} k$$

### Double Summation
$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$$

## Limits

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n = e^x$$

---

*This document showcases various mathematical notation and formulas that can be rendered in markdown using LaTeX syntax.*
`;

// Empty state
export const EMPTY_MD = '';