import type { BackendCapabilities, BackendCompat, BackendProtocol } from '$lib/types';

/** Prefix for generated ids of user-added backends. */
export const BACKEND_ID_PREFIX = 'backend';

/** Protocols a configured backend can speak, in display order. */
export const BACKEND_PROTOCOLS: readonly BackendProtocol[] = ['llama.cpp', 'openai'];

/** Chat completions path used when a backend does not override it. */
export const DEFAULT_BACKEND_CHAT_PATH = '/v1/chat/completions';

/** Models listing path used when a backend does not override it. */
export const DEFAULT_BACKEND_MODELS_PATH = '/v1/models';

/** Id of the built-in backend that points at the server serving this UI. */
export const LOCAL_BACKEND_ID = 'local';

/** Capabilities of a full llama.cpp server. */
const LLAMA_CPP_CAPABILITIES: BackendCapabilities = {
	corsProxy: true,
	loadUnload: true,
	props: true,
	resumableStreams: true,
	router: true,
	slots: true,
	statusFeed: true,
	tools: true
};
/** Capabilities of a plain OpenAI-compatible endpoint. */
const COMPATIBLE_CAPABILITIES: BackendCapabilities = {
	corsProxy: false,
	loadUnload: false,
	props: false,
	resumableStreams: false,
	router: false,
	slots: false,
	statusFeed: false,
	tools: false
};

/** Capabilities per backend protocol. */
export const BACKEND_CAPABILITIES: Record<BackendProtocol, BackendCapabilities> = {
	'llama.cpp': LLAMA_CPP_CAPABILITIES,
	openai: COMPATIBLE_CAPABILITIES
};

/** Default wire quirks per protocol. */
export const BACKEND_COMPAT: Record<BackendProtocol, BackendCompat> = {
	// llama-server reports its own timings, so it needs no usage chunk
	'llama.cpp': { maxTokensField: 'max_tokens', supportsUsageInStreaming: false },
	openai: { maxTokensField: 'max_tokens', supportsUsageInStreaming: true }
};

/**
 * Fields that may carry a model's context size in an OpenAI-compatible model
 * listing. Providers pick their own name, and most report nothing at all.
 */
export const MODEL_CONTEXT_LENGTH_FIELDS = [
	'context_length',
	'context_window',
	'max_context_length',
	'max_position_embeddings'
] as const;

/** Favicon extract keyed by domain, for backends with no bundled mark. */
export const FAVICON_SERVICE_URL = 'https://www.google.com/s2/favicons?domain=';
