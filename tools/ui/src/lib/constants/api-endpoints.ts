export const API_MODELS = {
	LIST: '/v1/models',
	LOAD: '/models/load',
	UNLOAD: '/models/unload',
	SSE: '/models/sse'
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const API_CHAT = {
	COMPLETIONS: './v1/chat/completions',
	CONTROL: './v1/chat/completions/control'
};

// slot introspection, requires the --slots flag on the server
export const API_SLOTS = {
	LIST: './slots'
};

export const API_TOOLS = {
	LIST: '/tools',
	EXECUTE: '/tools'
};

// resumable stream routes, the conv::model identity is appended as a path segment
export const API_STREAM = {
	BASE: './v1/stream',
	LOOKUP: './v1/streams/lookup'
};

/** CORS proxy endpoint path */
export const CORS_PROXY_ENDPOINT = '/cors-proxy';
