const base = import.meta.env.VITE_PUBLIC_SERVER_ORIGIN || '';

export const API_KEY = import.meta.env.VITE_API_KEY || '';

export const API_MODELS = {
	LIST: base + '/v1/models',
	LOAD: base + '/models/load',
	UNLOAD: base + '/models/unload'
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const API_CHAT = {
	COMPLETIONS: base + '/v1/chat/completions',
	CONTROL: base + '/v1/chat/completions/control'
};

// slot introspection, requires the --slots flag on the server
export const API_SLOTS = {
	LIST: base + '/slots'
};

export const API_TOOLS = {
	LIST: base + '/tools',
	EXECUTE: base + '/tools'
};

export const API_PROPS = {
	LIST: base + '/props'
};

/** CORS proxy endpoint path */
export const CORS_PROXY_ENDPOINT = base + '/cors-proxy';
