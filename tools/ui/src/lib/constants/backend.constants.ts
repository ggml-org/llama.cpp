import type {
	BackendCapabilities,
	BackendCompat,
	BackendPreset,
	BackendProtocol
} from '$lib/types';

/** Version sent with the Anthropic Messages API. */
export const ANTHROPIC_API_VERSION = '2023-06-01';

/** Prefix for generated ids of user-added backends. */
export const BACKEND_ID_PREFIX = 'backend';

/** Protocols a configured backend can speak, in display order. */
export const BACKEND_PROTOCOLS: readonly BackendProtocol[] = ['llama.cpp', 'openai', 'anthropic'];

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
/** Capabilities of a plain OpenAI- or Anthropic-compatible endpoint. */
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
	anthropic: COMPATIBLE_CAPABILITIES,
	'llama.cpp': LLAMA_CPP_CAPABILITIES,
	openai: COMPATIBLE_CAPABILITIES
};

/** Default wire quirks per protocol. */
export const BACKEND_COMPAT: Record<BackendProtocol, BackendCompat> = {
	// the Messages API has no OpenAI-style token cap or usage-in-stream toggle
	anthropic: { maxTokensField: 'max_tokens', supportsUsageInStreaming: false },
	// llama-server reports its own timings, so it needs no usage chunk
	'llama.cpp': { maxTokensField: 'max_tokens', supportsUsageInStreaming: false },
	openai: { maxTokensField: 'max_tokens', supportsUsageInStreaming: true }
};

/**
 * Ready-made endpoints offered when adding a backend. `custom` intentionally
 * carries no URL so the user starts from an empty form.
 */
export const BACKEND_PRESETS: readonly BackendPreset[] = [
	{
		baseUrl: 'https://router.huggingface.co',
		iconUrl: '/backend-presets/huggingface.svg',
		id: 'huggingface',
		name: 'Hugging Face',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://openrouter.ai/api',
		// v2 brand: purple on light, lime on dark
		iconUrlDark: '/backend-presets/openrouter-dark.svg',
		iconUrlLight: '/backend-presets/openrouter-light.svg',
		id: 'openrouter',
		name: 'OpenRouter',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
		chatPath: '/chat/completions',
		iconUrl: '/backend-presets/google.svg',
		id: 'google',
		modelsPath: '/models',
		name: 'Google',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.z.ai/api/paas/v4',
		chatPath: '/chat/completions',
		iconUrl: '/backend-presets/zai.png',
		id: 'zai',
		modelsPath: '/models',
		name: 'Z.ai',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.deepseek.com',
		iconUrl: '/backend-presets/deepseek.svg',
		id: 'deepseek',
		name: 'DeepSeek',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.moonshot.ai',
		iconUrl: '/backend-presets/kimi.png',
		id: 'kimi',
		name: 'Kimi',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
		chatPath: '/chat/completions',
		iconUrl: '/backend-presets/qwen.svg',
		id: 'qwen',
		modelsPath: '/models',
		name: 'Qwen',
		protocol: 'openai'
	},
	{
		baseUrl: '',
		id: 'custom',
		name: 'Custom',
		protocol: 'openai'
	}
];
