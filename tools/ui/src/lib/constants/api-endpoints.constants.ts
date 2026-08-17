export const API_MODELS = {
	LIST: '/v1/models',
	LOAD: '/models/load',
	SSE: '/models/sse',
	UNLOAD: '/models/unload'
};

// Chat completion and control routes.
export const API_CHAT = {
	COMPLETIONS: './v1/chat/completions',
	CONTROL: './v1/chat/completions/control'
};

// Slot introspection; requires the server `--slots` flag.
export const API_SLOTS = {
	LIST: './slots'
};

export const API_TOOLS = {
	EXECUTE: '/tools',
	LIST: '/tools'
};

export const API_SKILLS = {
	LIST: '/skills',
	READ: '/skills/read'
};

/** Tokenizer route used by Skills catalog packing. */
export const API_TOKENIZE = '/tokenize';

// Resumable stream routes and retry delay while the model loads.
export const STREAM_RESUME_RETRY_MS = 2000;

export const API_STREAM = {
	BASE: './v1/stream',
	LOOKUP: './v1/streams/lookup'
};

/** CORS proxy route. */
export const CORS_PROXY_ENDPOINT = '/cors-proxy';
