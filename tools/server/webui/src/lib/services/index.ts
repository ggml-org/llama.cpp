export { ChatService } from './chat.service';
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
export { ParameterSyncService } from './parameter-sync.service';
export { MCPService } from './mcp.service';
