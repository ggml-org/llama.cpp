/** Query params the chat routes read from the URL. */
export const URL_PARAMS = {
	/** Load the selected model instead of waiting for the first message. */
	LOAD: 'load',
	/** Model to select. */
	MODEL: 'model',
	/** Prompt to send on arrival. */
	QUERY: 'q'
} as const;

export const ROUTES = {
	/** Chat base — for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** Model detail - for dynamic model URLs use RouterService. */
	MANAGE_MODEL: '#/models-hub/[modelId]',
	/** Model hub - browse and download HuggingFace GGUF models. */
	MANAGE_MODELS: '#/models-hub',
	/** Model manager - installed models from /v1/models. */
	MODEL_MANAGER: '#/model-manager',
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** Search — mobile-only full-page conversation search. */
	SEARCH: '#/search',
	/** Root — start of the app. */
	START: '#/'
} as const;
