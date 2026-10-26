# llama-ui

A modern, feature-rich web interface for llama-server built with SvelteKit. This UI provides an intuitive chat interface with advanced file handling, conversation management, and comprehensive model interaction capabilities.

Llama UI supports two server operation modes:

- **MODEL mode** - Single model operation (standard llama-server)
- **ROUTER mode** - Multi-model operation with dynamic model loading/unloading

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Build Pipeline](#build-pipeline)
- [Architecture](#architecture)
- [Data Flows](#data-flows)
- [Architectural Patterns](#architectural-patterns)
- [Testing](#testing)

---

## Features

### Chat Interface

- **Streaming responses** with real-time updates
- **Reasoning content** - Support for models with thinking/reasoning blocks
- **Dark/light theme** with system preference detection
- **Responsive design** for desktop and mobile

### File Attachments

- **Images** - JPEG, PNG, GIF, WebP, SVG (with PNG conversion)
- **Documents** - PDF (text extraction or image conversion for vision models)
- **Audio** - MP3, WAV for audio-capable models
- **Text files** - Source code, markdown, and other text formats
- **Drag-and-drop** and paste support with rich previews

### Conversation Management

- **Branching** - Branch messages conversations at any point by editing messages or regenerating responses, navigate between branches
- **Regeneration** - Regenerate responses with optional model switching (ROUTER mode)
- **Import/Export** - JSON format for backup and sharing
- **Search** - Find conversations by title or content

### Advanced Rendering

- **Syntax highlighting** - Code blocks with language detection
- **Math formulas** - KaTeX rendering for LaTeX expressions
- **Markdown** - Full GFM support with tables, lists, and more

### Multi-Model Support (ROUTER mode)

- **Models manager** - one table for every model the server can serve, split into loaded, downloading, favorites, local and hidden sections, with search, context/capability/modality filters, sorting, load/unload, delete from disk and hide. It opens from the sidebar, from a model row or from the selector.
- **Model selector** with Loaded/Available groups
- **Automatic loading** - Models load on selection
- **Modality validation** - Prevents sending images to non-vision models
- **LRU unloading** - Server auto-manages model cache
- **Downloads** - In-flight downloads stay visible above the selector and can be paused, resumed or cancelled from the manager
- **Hub metadata** - Avatars, context length and chat template come from the Hugging Face Hub only while the `Use Hugging Face Hub API for models metadata` setting is on. It is off by default, and the UI then shows what the server reports for `/v1/models`

### Keyboard Shortcuts

| Shortcut           | Action               |
| ------------------ | -------------------- |
| `Shift+Ctrl/Cmd+O` | New chat             |
| `Shift+Ctrl/Cmd+E` | Edit conversation    |
| `Shift+Ctrl/Cmd+D` | Delete conversation  |
| `Ctrl/Cmd+K`       | Search conversations |
| `Ctrl/Cmd+B`       | Toggle sidebar       |

### Developer Experience

- **Request tracking** - Monitor token generation with `/slots` endpoint
- **Storybook** - Component library with visual testing
- **Hot reload** - Instant updates during development

---

## Getting Started

### Prerequisites

- **Node.js** 18+ (20+ recommended)
- **npm** 9+
- **llama-server** running locally (for API access)

### 1. Install Dependencies

```bash
cd tools/ui
npm ci
```

### 2. Start llama-server

In a separate terminal, start the backend server:

```bash
# Single model (MODEL mode)
./llama-server -m model.gguf

# Multi-model (ROUTER mode)
./llama-server --models-dir /path/to/models
```

### 3. Start Development Servers

```bash
npm run dev
```

This starts:

- **Vite dev server** at `http://localhost:5173` - The main UI frontend app
- **Storybook** at `http://localhost:6006` - Component documentation

The Vite dev server proxies API requests to `SERVER_ORIGIN` (with fallback to default llama-server `9931` port):

```typescript
// vite.config.ts proxy configuration
proxy: {
	'/v1': SERVER_ORIGIN,
	'/props': SERVER_ORIGIN,
	'/models': SERVER_ORIGIN,
	'/tools': SERVER_ORIGIN,
	'/slots': SERVER_ORIGIN,
	'/cors-proxy': SERVER_ORIGIN
},
```

### Development Workflow

1. Open `http://localhost:5173` in your browser
2. Make changes to `.svelte`, `.ts`, or `.css` files
3. Changes hot-reload instantly
4. Use Storybook at `http://localhost:6006` for isolated component development

---

## Tech Stack

| Layer             | Technology                      | Purpose                                                  |
| ----------------- | ------------------------------- | -------------------------------------------------------- |
| **Framework**     | SvelteKit + Svelte 5            | Reactive UI with runes (`$state`, `$derived`, `$effect`) |
| **UI Components** | shadcn-svelte + bits-ui         | Accessible, customizable component library               |
| **Drawers**       | vaul-svelte                     | Draggable drawer overlays (mobile pickers, model pane)   |
| **Styling**       | TailwindCSS 4                   | Utility-first CSS with design tokens                     |
| **Database**      | IndexedDB (Dexie)               | Client-side storage for conversations and messages       |
| **Build**         | Vite                            | Fast bundling with static adapter                        |
| **Testing**       | Playwright + Vitest + Storybook | E2E, unit, and visual testing                            |
| **Markdown**      | remark + rehype                 | Markdown processing with KaTeX and syntax highlighting   |

### Key Dependencies

```json
{
	"svelte": "^5.0.0",
	"bits-ui": "^2.8.11",
	"dexie": "^4.0.11",
	"pdfjs-dist": "^5.4.54",
	"highlight.js": "^11.11.1",
	"rehype-katex": "^7.0.1"
}
```

---

## Build Pipeline

### Development Build

```bash
npm run dev
```

Runs Vite in development mode with:

- Hot Module Replacement (HMR)
- Source maps
- Proxy to llama-server

### Production Build

```bash
npm run build
```

The build process:

1. **Vite Build** - Bundles all TypeScript, Svelte, and CSS
2. **Static Adapter** - Outputs to `../../build/tools/ui/dist` (llama-server's static file directory)
3. **Post-Build Script** - Cleans up intermediate files
4. **Custom Plugin** - Creates `index.html` with:
   - Inlined favicon as base64
   - GZIP compression (level 9)
   - Deterministic output (zeroed timestamps)

```text
tools/ui/        →  build  →  build/tools/ui/dist/
├── src/                                 ├── index.html  (served by llama-server)
├── static/                              └── (favicon inlined)
└── ...
```

### SvelteKit Configuration

```javascript
// svelte.config.js
adapter: adapter({
  pages: '../../build/tools/ui/dist',      // Output directory
  assets: '../../build/tools/ui/dist',     // Static assets
  fallback: 'index.html',  // SPA fallback
  strict: true
}),
output: {
  bundleStrategy: 'inline' // Single-file bundle
}
```

### Integration with llama-server

llama-ui is embedded directly into the llama-server binary:

1. `npm run build` outputs `index.html` to `build/tools/ui/dist/`
2. llama-server compiles this into the binary at build time
3. When accessing `/`, llama-server serves the bundled HTML

This results in a **single portable binary** with the full Llama UI included.

---

## Architecture

Llama UI follows a layered architecture with unidirectional data flow:

```text
Routes → Components → Hooks → Stores → Services → Storage/API
```

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Routes["📍 Routes"]
        R1["/ (Welcome)"]
        R2["/chat/[id]"]
        R3["/mcp-servers"]
        R4["/search"]
        R5["/settings"]
        RL["+layout.svelte"]
    end

    subgraph Components["🧩 Components"]
        C_Screen["ChatScreen"]
        C_Form["ChatForm"]
        C_Messages["ChatMessages"]
        C_Sidebar["ChatSidebar"]
        C_Models["ModelsSelector"]
        C_Settings["ChatSettings"]
        C_Mcp["McpServers"]
    end

    subgraph Hooks["🔌 Hooks"]
        H1["use-chat-screen-active-model"]
        H2["use-processing-state"]
        H3["use-context-gauge"]
        H4["use-models-selector"]
        H5["use-tools-panel"]
    end

    subgraph Stores["🗄️ Stores"]
        S1["chatStore"]
        S2["conversationsStore"]
        S3["modelsStore"]
        S4["mcpStore"]
        S5["agenticStore"]
        S6["serverStore"]
        S7["settingsStore"]
        S8["toolsStore"]
    end

    subgraph Services["⚙️ Services"]
        SV1["ChatService"]
        SV2["ModelsService"]
        SV3["PropsService"]
        SV4["DatabaseService"]
        SV5["MCPService"]
        SV6["ToolsService"]
        SV7["SandboxService"]
    end

    subgraph Storage["💾 Storage"]
        ST1["IndexedDB"]
        ST2["LocalStorage"]
    end

    subgraph APIs["🌐 llama-server"]
        API1["/v1/chat/completions"]
        API2["/props"]
        API3["/models/*"]
        API4["/tools"]
    end

    R1 & R2 --> C_Screen
    RL --> C_Sidebar
    C_Screen --> C_Form & C_Messages & C_Settings
    C_Screen --> H1 & H2 & H3
    C_Models --> H4
    C_Mcp --> S4
    C_Screen --> S1 & S2 & S3
    C_Models --> S3
    H1 --> S3
    S1 --> SV1 & SV4
    S2 --> SV4
    S3 --> SV2 & SV3
    S4 --> SV5
    S5 --> SV1 & SV5 & SV6 & SV7
    SV4 --> ST1
    SV1 --> API1
    SV2 --> API3
    SV3 --> API2
    SV6 --> API4
```

### Layer Breakdown

#### Routes (`src/routes/`)

- **`/`** - Welcome screen, creates new conversation
- **`/chat/[id]`** - Active chat interface
- **`/mcp-servers`** - MCP server management
- **`/search`** - Conversation search
- **`/settings`** - Settings (optional `[[section]]`)
- **`+layout.svelte`** - Sidebar, navigation, global initialization

#### Components (`src/lib/components/`)

Components are organized in `app/` (application-specific) and `ui/` (shadcn-svelte primitives).

**Chat Components** (`app/chat/`):

| Component          | Responsibility                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| `ChatScreen/`      | Main chat container, coordinates message list, input form, and attachments  |
| `ChatForm/`        | Message input textarea with file upload, paste handling, keyboard shortcuts |
| `ChatMessages/`    | Message list with branch navigation, regenerate/continue/edit actions       |
| `ChatAttachments/` | File attachment previews, drag-and-drop, PDF/image/audio handling           |
| `ChatSettings/`    | Parameter sliders (temperature, top-p, etc.) with server default sync       |
| `ChatSidebar/`     | Conversation list, search, import/export, navigation                        |

**Dialog Components** (`app/dialogs/`):

| Component                      | Responsibility                                           |
| ------------------------------ | -------------------------------------------------------- |
| `DialogChatSettings`           | Full-screen settings configuration                       |
| `DialogManageModels`           | Models manager: browse, load, download and delete models |
| `DialogModelInformation`       | Model details (context size, modalities, parallel slots) |
| `DialogChatAttachmentsPreview` | Full preview for images, PDFs (text or page view), code  |
| `DialogConfirmation`           | Generic confirmation for destructive actions             |
| `DialogConversationRename`     | Edit conversation title                                  |

**Server/Model Components** (`app/server/`, `app/models/`):

| Component                           | Responsibility                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `ServerErrorSplash`                 | Error display when server is unreachable                                                                      |
| `ModelsSelector/`                   | Model dropdown and mobile sheet: favorites, loaded and local sections, rows windowed and folded into families |
| `ModelsManager/`                    | Models manager table and model pane: sections, filters, sorting, quants folded per repo, downloads            |
| `ModelAvatar`                       | Org avatar of a model, with the quantizer badge; follows the family grouping of the list                      |
| `ModelId`, `ModelBadge`             | Model name, aliases, tags, quantization and draft sidecar badges                                              |
| `ModelContext`, `ModelCapabilities` | Context window and capability icons, filled from the Hub when it is enabled                                   |

**Shared UI Components** (`app/misc/`):

| Component                                 | Responsibility                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `MarkdownContent`                         | Markdown rendering with KaTeX, syntax highlighting, copy buttons        |
| `SyntaxHighlightedCode`                   | Code blocks with language detection and highlighting                    |
| `ActionIcon`, `DropdownMenuActions`       | Reusable action button and action menu                                  |
| `BadgesModality`, `BadgeInfo`             | Status and capability badges                                            |
| `CollapsibleSection`, `CollapsibleRegion` | Section header with a trigger that expands in place                     |
| `GroupedList`                             | Grouped rows with foldable groups, per-group windows and show-more rows |
| `ScrollCarousel`, `TruncatedText`         | Horizontal scroll strip and single-line truncation with tooltip         |

#### Hooks (`src/lib/hooks/`)

Hooks are the thin view-layer between components and stores: they own UI concerns (scroll, drag-and-drop, keyboard shortcuts, pickers, selection) and translate store state into view state.

| Hook                            | Responsibility                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| `use-chat-screen-active-model`  | Active model resolution + modality capability detection                                         |
| `use-processing-state`          | View over `chatStore.processing` for streaming progress/tokens                                  |
| `use-context-gauge`             | View over `contextStatsStore` for the context usage gauge                                       |
| `use-models-selector`           | Model selector state: hidden models, family grouping, windowed rows, selection and load on pick |
| `use-tools-panel`               | Tools panel state                                                                               |
| `use-reasoning-menu`            | Reasoning-effort menu state                                                                     |
| `use-attachment-menu`           | Attachment menu + modality flags                                                                |
| `use-draft-messages`            | Per-chat draft message/files persistence                                                        |
| `use-chat-form-pickers`         | Chat form pickers (commands, mentions)                                                          |
| `use-debounced-search`          | Shared debounced async search for pickers                                                       |
| `use-picker-navigation`         | Picker keyboard navigation                                                                      |
| `use-chat-message-edit-context` | Message edit context (content + extras)                                                         |
| `use-chat-screen-drag-and-drop` | Drag-and-drop state machine                                                                     |
| `use-chat-screen-file-upload`   | File upload queue + capability validation                                                       |
| `use-chat-screen-scroll`        | Scroll container binding + navigation guard                                                     |
| `use-auto-scroll`               | Auto-scroll controller for streaming                                                            |
| `use-marquee-selection`         | Shift+click / marquee range selection                                                           |
| `use-keyboard-shortcuts`        | Global keyboard shortcuts                                                                       |
| `use-settings-navigation`       | Settings section navigation                                                                     |
| `use-pwa`                       | PWA install/update + version mismatch detection                                                 |

#### Stores (`src/lib/stores/`)

Stores own reactive application state as Svelte 5 runes. Larger stores are split into directories and compose focused sub-stores behind a narrow host interface (see Architectural Patterns).

| Store                | Responsibility                                                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `chatStore`          | Chat lifecycle, streaming, abort control, error handling; composes `processing`, `activity`, `streams`, `flows`                           |
| `conversationsStore` | Conversation CRUD, message branching, navigation, import/export; composes `preferences`                                                   |
| `modelsStore`        | Model list, selection, loading/unloading (ROUTER), recent picks, favorites, hidden models and list open state; composes `props`, `status` |
| `mcpStore`           | MCP host role: multi-server lifecycle, tool routing, server icons; composes `health`, `resources`                                         |
| `agenticStore`       | Multi-turn agentic loop orchestration, tool execution; composes `gates`                                                                   |
| `serverStore`        | Server connection state, `/props`, role detection, modalities                                                                             |
| `settingsStore`      | User preferences, theme, parameter sync with server defaults                                                                              |
| `toolsStore`         | Tool registry: server + MCP tools, enabled set for the LLM                                                                                |
| `permissionsStore`   | Persisted tool permission grants                                                                                                          |
| `contextStatsStore`  | Context window usage for the active conversation                                                                                          |
| `draftMessagesStore` | Per-chat draft message/files                                                                                                              |
| `deviceStore`        | Browser environment signals (mobile, OS, theme)                                                                                           |
| `versionStore`       | Build version information                                                                                                                 |

#### Services (`src/lib/services/`)

Services are a stateless protocol layer: static methods, pure I/O, no reactive state. Stores consume them for all API and storage access.

| Service                       | Responsibility                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `ChatService`                 | `/v1/chat/completions` streaming + SSE parsing, message format conversion                                                     |
| `ModelsService`               | `/models`, `/models/load`, `/models/unload`                                                                                   |
| `PropsService`                | `/props`, `/props?model=`                                                                                                     |
| `HuggingFaceService`          | Hugging Face Hub metadata: model details, avatars, README and file tree, cached per repo and off unless the Hub setting is on |
| `DatabaseService`             | IndexedDB operations via Dexie                                                                                                |
| `MCPService`                  | MCP protocol: transports, connect, list/execute tools, prompts, resources                                                     |
| `ToolsService`                | Server tool list/execute/stream (`/tools`)                                                                                    |
| `SandboxService`              | Browser JS execution in a sandboxed worker                                                                                    |
| `ParameterSyncService`        | Syncs settings with server defaults                                                                                           |
| `ConversationTransferService` | Conversation import/export JSONL + ZIP format                                                                                 |
| `MigrationService`            | Non-destructive localStorage/IndexedDB migrations                                                                             |
| `RouterService`               | Dynamic route URL construction                                                                                                |

---

## Data Flows

### MODEL Mode (Single Model)

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant Stores
    participant DB as IndexedDB
    participant API as llama-server

    Note over User,API: Initialization
    UI->>Stores: initStores() (awaited by route loads)
    Stores->>Stores: run migrations
    Stores->>DB: load conversations (background)
    Stores->>API: GET /props
    API-->>Stores: server config
    Stores->>API: GET /v1/models
    API-->>Stores: single model (auto-selected)

    Note over User,API: Chat Flow
    User->>UI: send message
    Stores->>DB: save user message
    Stores->>API: POST /v1/chat/completions (stream)
    loop streaming
        API-->>Stores: SSE chunks
        Stores-->>UI: reactive update
    end
    Stores->>DB: save assistant message
```

### ROUTER Mode (Multi-Model)

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant Stores
    participant API as llama-server

    Note over User,API: Initialization
    Stores->>API: GET /props
    API-->>Stores: {role: "router"}
    Stores->>API: GET /models
    API-->>Stores: models[] with status

    Note over User,API: Model Selection
    User->>UI: select model
    alt model not loaded
        Stores->>API: POST /models/load
        loop poll status
            Stores->>API: GET /models
        end
        Stores->>API: GET /props?model=X
    end
    Stores->>Stores: validate modalities

    Note over User,API: Chat Flow
    Stores->>API: POST /v1/chat/completions {model: X}
    loop streaming
        API-->>Stores: SSE chunks + model info
    end
```

---

## Architectural Patterns

### 1. Reactive State with Svelte 5 Runes

All stores use Svelte 5's fine-grained reactivity:

```typescript
// Store with reactive state
class ChatStore {
	#isLoading = $state(false);
	#currentResponse = $state('');

	// Derived values auto-update
	get isStreaming() {
		return $derived(this.#isLoading && this.#currentResponse.length > 0);
	}
}

// Exported reactive accessors
export const isLoading = () => chatStore.isLoading;
export const currentResponse = () => chatStore.currentResponse;
```

### 2. Unidirectional Data Flow

Data flows in one direction, making state predictable:

```mermaid
flowchart LR
    subgraph UI["UI Layer"]
        A[User Action] --> B[Component]
    end

    subgraph State["State Layer"]
        B --> C[Store Method]
        C --> D[State Update]
    end

    subgraph IO["I/O Layer"]
        C --> E[Service]
        E --> F[API / IndexedDB]
        F -.->|Response| D
    end

    D -->|Reactive| B
```

Components dispatch actions to stores, stores coordinate with services for I/O, and state updates reactively propagate back to the UI.

### 3. Per-Conversation State

Enables concurrent streaming across multiple conversations. Loading is tracked
per conversation by the activity ledger (`chatStore.activity`), while streaming
state and abort controllers live in per-conversation maps:

```typescript
class ChatStore {
	chatStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	abortControllers = new SvelteMap<string, AbortController>();
}
```

### 4. Message Branching with Tree Structure

Conversations are stored as a tree, not a linear list:

```typescript
interface DatabaseMessage {
	id: string;
	parent: string | null; // Points to parent message
	children: string[]; // List of child message IDs
	// ...
}

interface DatabaseConversation {
	currentNode: string; // Currently viewed branch tip
	// ...
}
```

Navigation between branches updates `currentNode` without losing history.

### 5. Layered Service Architecture

Stores handle state; services handle I/O:

```text
┌─────────────────┐
│     Stores      │  Business logic, state management
├─────────────────┤
│    Services     │  API calls, database operations
├─────────────────┤
│   Storage/API   │  IndexedDB, LocalStorage, HTTP
└─────────────────┘
```

### 6. Server Role Abstraction

Single codebase handles both MODEL and ROUTER modes:

```typescript
// serverStore.ts
get isRouterMode() {
  return this.role === ServerRole.ROUTER;
}

// Components conditionally render based on mode
{#if isRouterMode()}
  <ModelsSelector />
{/if}
```

### 7. Modality Validation

Prevents sending attachments to incompatible models. The
`use-chat-screen-active-model` hook derives the active model's capabilities
from `modelsStore.props`:

```typescript
// use-chat-screen-active-model hook
const hasVisionModality = $derived.by(() => modelsStore.props.modelSupportsVision(activeModelId));
const hasAudioModality = $derived.by(() => modelsStore.props.modelSupportsAudio(activeModelId));
```

### 8. Persistent Storage Strategy

Data is persisted across sessions using two storage mechanisms:

```mermaid
flowchart TB
    subgraph Browser["Browser Storage"]
        subgraph IDB["IndexedDB (Dexie)"]
            C[Conversations]
            M[Messages]
        end
        subgraph LS["LocalStorage"]
            S[Settings Config]
            O[User Overrides]
            T[Theme Preference]
        end
    end

    subgraph Stores["Svelte Stores"]
        CS[conversationsStore] --> C
        CS --> M
        SS[settingsStore] --> S
        SS --> O
        SS --> T
    end
```

- **IndexedDB**: Conversations and messages (large, structured data)
- **LocalStorage**: Settings, user parameter overrides, theme, model favorites, recent picks, hidden models and model list open state (small key-value data)
- **Memory only**: Server props, model list (fetched fresh on each session), Hugging Face Hub details (cached per repo for the session)

### 9. Remote Assets Under Cross-Origin Isolation

llama-server serves the UI with `Cross-Origin-Embedder-Policy: require-corp` and `Cross-Origin-Opener-Policy: same-origin`, so a cross-origin subresource has to be allowed by the target origin. The UI handles that per asset kind:

- **Hub avatars** are requested in CORS mode (`crossorigin="anonymous"`), which the Hub allows
- **Hub metadata** is fetched with `fetch()`, which is a CORS request by default, and only while the Hub setting is on
- **MCP server icons and web-search result favicons** have no CORS guarantee, so they go through the server's `/cors-proxy` when the user started llama-server with `--ui-mcp-proxy`; without it they fall back to the bundled glyph
- **Per-row Hub lookups** wait until the row is near the viewport (the `nearViewport` action), so a long model list does not fire one request per row on mount

---

## Testing

### Test Types

| Type          | Tool               | Location         | Command             |
| ------------- | ------------------ | ---------------- | ------------------- |
| **Unit**      | Vitest             | `tests/unit/`    | `npm run test:unit` |
| **UI/Visual** | Storybook + Vitest | `tests/stories/` | `npm run test:ui`   |
| **E2E**       | Playwright         | `tests/e2e/`     | `npm run test:e2e`  |
| **Client**    | Vitest             | `tests/client/`. | `npm run test:unit` |

### Running Tests

```bash
# All tests
npm run test

# Individual test suites
npm run test:e2e      # End-to-end (requires llama-server)
npm run test:client   # Client-side unit tests
npm run test:server   # Server-side unit tests
npm run test:ui       # Storybook visual tests
```

### Storybook Development

```bash
npm run storybook     # Start Storybook dev server on :6006
npm run build-storybook  # Build static Storybook
```

### Linting and Formatting

```bash
npm run lint          # Check code style
npm run format        # Auto-format with Prettier
npm run check         # TypeScript type checking
```

---

## Project Structure

```text
tools/ui/
├── src/
│   ├── lib/
│   │   ├── components/   # UI components (app/, ui/)
│   │   ├── hooks/        # Svelte hooks
│   │   ├── stores/       # State management
│   │   ├── services/     # API and database services
│   │   ├── types/        # TypeScript interfaces
│   │   └── utils/        # Utility functions
│   ├── routes/           # SvelteKit routes
│   └── styles/           # Global styles
├── static/               # Static assets
├── tests/                # Test files
└── .storybook/           # Storybook configuration
```

---

## Related Documentation

- [llama.cpp Server README](../server/README.md) - Full server documentation
- [Multimodal Documentation](../../docs/multimodal.md) - Image and audio support
- [Function Calling](../../docs/function-calling.md) - Tool use capabilities
