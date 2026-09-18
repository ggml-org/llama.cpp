/**
 * Backend types.
 *
 * A backend is one API endpoint the UI can talk to. The built-in `local`
 * backend is the llama-server serving the UI. External backends are
 * user-configured endpoints that speak an OpenAI-compatible
 * protocol.
 */

/** Request/response shape a backend speaks. */
export type BackendProtocol = 'llama.cpp' | 'openai';

/**
 * Wire-level quirks of a backend's protocol. Capabilities gate llama.cpp
 * features; compat describes how the request and stream payloads differ.
 */
export interface BackendCompat {
	/** Field carrying the output token cap. */
	maxTokensField: 'max_completion_tokens' | 'max_tokens';
	/** Whether the endpoint accepts stream_options.include_usage. */
	supportsUsageInStreaming: boolean;
}

/**
 * Features a backend supports. A llama.cpp server exposes extra endpoints on
 * top of the OpenAI-compatible API; plain OpenAI-compatible
 * endpoints only provide chat and model listing.
 */
export interface BackendCapabilities {
	/** llama-server's /cors-proxy endpoint for cross-origin MCP requests. */
	corsProxy: boolean;
	/** Router-mode model load/unload. */
	loadUnload: boolean;
	/** The /props endpoint with server role and generation defaults. */
	props: boolean;
	/** Resumable stream sessions (/v1/stream, /v1/streams/lookup). */
	resumableStreams: boolean;
	/** Multi-model router mode. */
	router: boolean;
	/** The /slots introspection endpoint. */
	slots: boolean;
	/** The /models/sse load and download progress feed. */
	statusFeed: boolean;
	/** The /tools listing and execution endpoint. */
	tools: boolean;
}

/**
 * One configured API endpoint.
 *
 * TODO: a backend paired by QR code is reached over WebRTC instead of plain
 * HTTP, so it needs a transport discriminator and its peer description here.
 */
export interface Backend {
	/** Bearer token / API key used for this backend. */
	apiKey?: string;
	/**
	 * API root the endpoint paths are appended to, e.g. https://api.example.com.
	 * Empty for the local backend, which resolves against the UI origin instead.
	 */
	baseUrl: string;
	/** Chat completions path override, e.g. /v1/messages. */
	chatPath?: string;
	/** Wire quirks overriding the protocol defaults. */
	compat?: Partial<BackendCompat>;
	/** Disabled backends stay configured but are not queried. */
	enabled: boolean;
	/** Extra headers merged into every request to this backend. */
	headers?: Record<string, string>;
	/** Stable identity. The local backend id is reserved. */
	id: string;
	/** Models listing path override, e.g. /models. */
	modelsPath?: string;
	name: string;
	protocol: BackendProtocol;
}

/** A ready-made backend configuration offered when adding a backend. */
export interface BackendPreset {
	/** Optional help text shown under the API key field. */
	apiKeyHelp?: string;
	baseUrl: string;
	/** One line describing the endpoint, shown on the preset card. */
	description?: string;
	chatPath?: string;
	/** Brand mark used in both themes, for logos that carry their own background. */
	iconUrl?: string;
	/** Brand mark for the dark theme. Preferred over `iconUrl` when paired with `iconUrlLight`. */
	iconUrlDark?: string;
	/** Brand mark for the light theme. Preferred over `iconUrl` when paired with `iconUrlDark`. */
	iconUrlLight?: string;
	/** Wire quirks this preset needs on top of the protocol defaults. */
	compat?: Partial<BackendCompat>;
	id: string;
	modelsPath?: string;
	name: string;
	protocol: BackendProtocol;
}
