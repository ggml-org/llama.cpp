import type { BackendPreset, BackendProtocol } from '$lib/types';

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

/**
 * Ready-made endpoints offered when adding a backend. `custom` intentionally
 * carries no URL so the user starts from an empty form.
 */
export const BACKEND_PRESETS: readonly BackendPreset[] = [
	{
		baseUrl: 'https://openrouter.ai/api',
		id: 'openrouter',
		name: 'OpenRouter',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.together.xyz',
		id: 'together',
		name: 'Together',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://router.huggingface.co',
		id: 'huggingface',
		name: 'Hugging Face',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.deepseek.com',
		id: 'deepseek',
		name: 'DeepSeek',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.moonshot.ai',
		id: 'kimi',
		name: 'Kimi',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.openai.com',
		id: 'openai',
		name: 'OpenAI',
		protocol: 'openai'
	},
	{
		baseUrl: 'https://api.anthropic.com',
		chatPath: '/v1/messages',
		id: 'anthropic',
		name: 'Anthropic',
		protocol: 'anthropic'
	},
	{
		baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
		chatPath: '/chat/completions',
		id: 'google',
		modelsPath: '/models',
		name: 'Google',
		protocol: 'openai'
	},
	{
		baseUrl: '',
		id: 'custom',
		name: 'Custom',
		protocol: 'openai'
	}
];
