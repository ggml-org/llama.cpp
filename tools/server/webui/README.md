# llama.cpp Web UI

A modern, feature-rich web interface for llama.cpp built with SvelteKit. This UI provides an intuitive chat interface with advanced file handling, conversation management, and comprehensive model interaction capabilities.

## Features

- **Modern Chat Interface** - Clean, responsive design with dark/light mode
- **File Attachments** - Support for images, text files, PDFs, and audio with rich previews and drag-and-drop support
- **Conversation Management** - Create, edit, branch, and search conversations
- **Advanced Markdown** - Code highlighting, math formulas (KaTeX), and content blocks
- **Reasoning Content** - Support for models with thinking blocks
- **Keyboard Shortcuts** - Keyboard navigation (Shift+Ctrl/Cmd+O for new chat, Shift+Ctrl/Cmdt+E for edit conversation, Shift+Ctrl/Cmdt+D for delete conversation, Ctrl/Cmd+K for search, Ctrl/Cmd+V for paste, Ctrl/Cmd+B for opening/collapsing sidebar)
- **Request Tracking** - Monitor processing with slots endpoint integration
- **UI Testing** - Storybook component library with automated tests

## Development

Install dependencies:

```bash
npm install
```

Start the development server + Storybook:

```bash
npm run dev
```

This will start both the SvelteKit dev server and Storybook on port 6006.

## Building

Create a production build:

```bash
npm run build
```

The build outputs static files to `../public` directory for deployment with llama.cpp server.

## Testing

Run the test suite:

```bash
# E2E tests
npm run test:e2e

# Unit tests
npm run test:unit

# UI tests
npm run test:ui

# All tests
npm run test
```

## Architecture

- **Framework**: SvelteKit with Svelte 5 runes
- **Components**: ShadCN UI + bits-ui design system
- **Database**: IndexedDB with Dexie for local storage
- **Build**: Static adapter for deployment with llama.cpp server
- **Testing**: Playwright (E2E) + Vitest (unit) + Storybook (components)
