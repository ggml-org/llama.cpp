// README Content
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
