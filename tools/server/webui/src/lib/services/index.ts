/**
 *
 * SERVICES
 *
 * Stateless service layer for API communication and data operations.
 * Services handle protocol-level concerns (HTTP, WebSocket, MCP, IndexedDB)
 * without managing reactive state — that responsibility belongs to stores.
 *
 * **Design Principles:**
 * - All methods are static — no instance state
 * - Pure I/O operations (network requests, database queries)
 * - No Svelte runes or reactive primitives
 * - Error handling at the protocol level; business-level error handling in stores
 *
 * **Architecture (bottom to top):**
 * - **Services** (this layer): Stateless protocol communication
 * - **Stores**: Reactive state management consuming services
 * - **Components**: UI consuming stores
 *
 */

/**
 * **ChatService** - Chat Completions API communication layer
 *
 * Handles direct communication with the llama-server's `/v1/chat/completions` endpoint.
 * Provides streaming and non-streaming response parsing, message format conversion
 * (DatabaseMessage → API format), and request lifecycle management.
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. Ephemeral and
 *   runtime-focused — sending messages, receiving streaming responses, managing request lifecycles.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   Managed by conversationsStore, conversations persist across sessions.
 *
 * **Architecture & Relationships:**
 * - **ChatService** (this class): Stateless API communication layer
 *   - Handles HTTP requests/responses with the llama-server
 *   - Manages streaming and non-streaming response parsing
 *   - Converts database messages to API format (multimodal, tool calls)
 *   - Handles error translation with user-friendly messages
 *
 * - **chatStore**: Primary consumer — uses ChatService for all AI model communication
 * - **agenticStore**: Uses ChatService for multi-turn agentic loop streaming
 * - **conversationsStore**: Provides message context for API requests
 *
 * **Key Responsibilities:**
 * - Streaming response handling with real-time content/reasoning/tool-call callbacks
 * - Non-streaming response parsing with complete response extraction
 * - Database message to API format conversion (attachments, tool calls, multimodal)
 * - Tool call delta merging for incremental streaming aggregation
 * - Request parameter assembly (sampling, penalties, custom params)
 * - File attachment processing (images, PDFs, audio, text, MCP prompts/resources)
 * - Reasoning content stripping from prompt history to avoid KV cache pollution
 * - Error translation (network, timeout, server errors → user-friendly messages)
 *
 * @see chatStore in stores/chat.svelte.ts — primary consumer for chat state management
 * @see agenticStore in stores/agentic.svelte.ts — uses ChatService for agentic loop streaming
 * @see conversationsStore in stores/conversations.svelte.ts — provides message context
 */
export { ChatService } from './chat.service';

/**
 * **DatabaseService** - IndexedDB persistence layer via Dexie ORM
 *
 * Provides stateless data access for conversations and messages using IndexedDB.
 * Handles all low-level storage operations including branching tree structures,
 * cascade deletions, and transaction safety for multi-table operations.
 *
 * **Architecture & Relationships (bottom to top):**
 * - **DatabaseService** (this class): Stateless IndexedDB operations
 *   - Lowest layer — direct Dexie/IndexedDB communication
 *   - Pure CRUD operations without business logic
 *   - Handles branching tree structure (parent-child relationships)
 *   - Provides transaction safety for multi-table operations
 *
 * - **conversationsStore**: Reactive state management layer
 *   - Uses DatabaseService for all persistence operations
 *   - Manages conversation list, active conversation, and messages in memory
 *
 * - **chatStore**: Active AI interaction management
 *   - Uses conversationsStore for conversation context
 *   - Directly uses DatabaseService for message CRUD during streaming
 *
 * **Key Responsibilities:**
 * - Conversation CRUD (create, read, update, delete)
 * - Message CRUD with branching support (parent-child relationships)
 * - Root message and system prompt creation
 * - Cascade deletion of message branches (descendants)
 * - Transaction-safe multi-table operations
 * - Conversation import with duplicate detection
 *
 * **Database Schema:**
 * - `conversations`: id, lastModified, currNode, name
 * - `messages`: id, convId, type, role, timestamp, parent, children
 *
 * **Branching Model:**
 * Messages form a tree structure where each message can have multiple children,
 * enabling conversation branching and alternative response paths. The conversation's
 * `currNode` tracks the currently active branch endpoint.
 *
 * @see conversationsStore in stores/conversations.svelte.ts — reactive layer on top of DatabaseService
 * @see chatStore in stores/chat.svelte.ts — uses DatabaseService directly for message CRUD during streaming
 */
export { DatabaseService } from './database.service';

/**
 * **ModelsService** - Model management API communication
 *
 * Handles communication with model-related endpoints for both MODEL (single model)
 * and ROUTER (multi-model) server modes. Provides model listing, loading/unloading,
 * and status checking without managing any model state.
 *
 * **Architecture & Relationships:**
 * - **ModelsService** (this class): Stateless HTTP communication
 *   - Sends requests to model endpoints
 *   - Parses and returns typed API responses
 *   - Provides model status utility methods
 *
 * - **modelsStore**: Primary consumer — manages reactive model state
 *   - Calls ModelsService for all model API operations
 *   - Handles polling, caching, and state updates
 *
 * **Key Responsibilities:**
 * - List available models via OpenAI-compatible `/v1/models` endpoint
 * - Load/unload models via `/models/load` and `/models/unload` (ROUTER mode)
 * - Model status queries (loaded, loading)
 *
 * **Server Mode Behavior:**
 * - **MODEL mode**: Only `list()` is relevant — single model always loaded
 * - **ROUTER mode**: Full lifecycle — `list()`, `listRouter()`, `load()`, `unload()`
 *
 * **Endpoints:**
 * - `GET /v1/models` — OpenAI-compatible model list (both modes)
 * - `POST /models/load` — Load a model (ROUTER mode only)
 * - `POST /models/unload` — Unload a model (ROUTER mode only)
 *
 * @see modelsStore in stores/models.svelte.ts — primary consumer for reactive model state
 */
export { ModelsService } from './models.service';

/**
 * **PropsService** - Server properties and capabilities retrieval
 *
 * Fetches server configuration, model information, and capabilities from the `/props`
 * endpoint. Supports both global server props and per-model props (ROUTER mode).
 *
 * **Architecture & Relationships:**
 * - **PropsService** (this class): Stateless HTTP communication
 *   - Fetches server properties from `/props` endpoint
 *   - Handles authentication and request parameters
 *   - Returns typed `ApiLlamaCppServerProps` responses
 *
 * - **serverStore**: Consumes global server properties (role detection, connection state)
 * - **modelsStore**: Consumes per-model properties (modalities, context size)
 * - **settingsStore**: Syncs default generation parameters from props response
 *
 * **Key Responsibilities:**
 * - Fetch global server properties (default generation settings, modalities)
 * - Fetch per-model properties in ROUTER mode via `?model=<id>` parameter
 * - Handle autoload control to prevent unintended model loading
 *
 * **API Behavior:**
 * - `GET /props` → Global server props (MODEL mode: includes modalities)
 * - `GET /props?model=<id>` → Per-model props (ROUTER mode: model-specific modalities)
 * - `&autoload=false` → Prevents model auto-loading when querying props
 *
 * @see serverStore in stores/server.svelte.ts — consumes global server props
 * @see modelsStore in stores/models.svelte.ts — consumes per-model props for modalities
 * @see settingsStore in stores/settings.svelte.ts — syncs default generation params from props
 */
export { PropsService } from './props.service';

/**
 * **ParameterSyncService** - Server defaults and user settings synchronization
 *
 * Manages the complex logic of merging server-provided default parameters with
 * user-configured overrides. Ensures the UI reflects the actual server state
 * while preserving user customizations. Tracks parameter sources (server default
 * vs user override) for display in the settings UI.
 *
 * **Architecture & Relationships:**
 * - **ParameterSyncService** (this class): Stateless sync logic
 *   - Pure functions for parameter extraction, merging, and diffing
 *   - No side effects — receives data in, returns data out
 *   - Handles floating-point precision normalization
 *
 * - **settingsStore**: Primary consumer — calls sync methods during:
 *   - Initial load (`syncWithServerDefaults`)
 *   - Settings reset (`forceSyncWithServerDefaults`)
 *   - Parameter info queries (`getParameterInfo`)
 *
 * - **PropsService**: Provides raw server props that feed into extraction
 *
 * **Key Responsibilities:**
 * - Extract syncable parameters from server `/props` response
 * - Merge server defaults with user overrides (user wins)
 * - Track parameter source (Custom vs Default) for UI badges
 * - Validate server parameter values by type (number, string, boolean)
 * - Create diffs between current settings and server defaults
 * - Floating-point precision normalization for consistent comparisons
 *
 * **Parameter Source Priority:**
 * 1. **User Override** (Custom badge) — explicitly set by user in settings
 * 2. **Server Default** (Default badge) — from `/props` endpoint
 * 3. **App Default** — hardcoded fallback when server props unavailable
 *
 * **Exports:**
 * - `ParameterSyncService` class — static methods for sync logic
 * - `SYNCABLE_PARAMETERS` — mapping of webui setting keys to server parameter keys
 *
 * @see settingsStore in stores/settings.svelte.ts — primary consumer for settings sync
 * @see ChatSettingsParameterSourceIndicator — displays parameter source badges in UI
 */
export { ParameterSyncService } from './parameter-sync.service';
