/**
 * Backend types.
 *
 * A backend is one API endpoint the UI can talk to. The built-in `local`
 * backend is the llama-server serving the UI. External backends are
 * user-configured endpoints that speak an OpenAI- or Anthropic-compatible
 * protocol.
 */

/** Request/response shape a backend speaks. */
export type BackendProtocol = 'llama.cpp' | 'openai' | 'anthropic';

/**
 * Features a backend supports. A llama.cpp server exposes extra endpoints on
 * top of the OpenAI-compatible API; plain OpenAI- and Anthropic-compatible
 * endpoints only provide chat and model listing.
 */
export interface BackendCapabilities {
	/** llama-server's /cors-proxy endpoint for cross-origin MCP requests. */
	corsProxy: boolean;
	/** Router-mode model load/unload. */
	loadUnload: boolean;
	/** The /props endpoint with server role and generation defaults. */
	props: boolean;
	/** Multi-model router mode. */
	router: boolean;
	/** The /slots introspection endpoint. */
	slots: boolean;
	/** The /models/sse load and download progress feed. */
	statusFeed: boolean;
	/** The /tools listing and execution endpoint. */
	tools: boolean;
}

/** One configured API endpoint. */
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
	chatPath?: string;
	id: string;
	modelsPath?: string;
	name: string;
	protocol: BackendProtocol;
}
