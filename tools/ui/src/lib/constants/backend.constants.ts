import type {
	BackendCapabilities,
	BackendCompat,
	BackendPreset,
	BackendProtocol
} from '$lib/types';

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

/** Favicon extract keyed by domain, for backends with no bundled mark. */
export const FAVICON_SERVICE_URL = 'https://www.google.com/s2/favicons?domain=';

/**
 * Ready-made endpoints offered when adding a backend.
 *
 * TODO: pair a llama.cpp server by scanning its QR code (WebRTC transport, so a
 * hosted PWA can reach a server on the user's network), see
 * ggml-org/llama.cpp#24577.
 */
export const BACKEND_PRESETS: readonly BackendPreset[] = [
	{
		baseUrl: 'https://router.huggingface.co',
		description: 'Open models from the Hub, served by inference providers.',
		iconUrl: '/backend-presets/huggingface.svg',
		id: 'huggingface',
		name: 'Hugging Face',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://openrouter.ai/api',
		description: 'Models from many providers behind a single API.',
		// v2 brand: purple on light, lime on dark
		iconUrlDark: '/backend-presets/openrouter-dark.svg',
		iconUrlLight: '/backend-presets/openrouter-light.svg',
		id: 'openrouter',
		name: 'OpenRouter',
		protocol: 'openai'
	}
];
