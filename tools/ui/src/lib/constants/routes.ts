/** Query params the chat routes read from the URL. */
export const URL_PARAMS = {
	/** Prompt to send on arrival. */
	QUERY: 'q',
	/** Model to select. */
	MODEL: 'model',
	/** Load the selected model instead of waiting for the first message. */
	LOAD: 'load',
	/** Start a new chat. */
	NEW_CHAT: 'new_chat'
} as const;

/** Settings section slugs — used for routes and navigation. */
export const SETTINGS_SECTION_SLUGS = {
	GENERAL: 'general',
	DISPLAY: 'display',
	SAMPLING: 'sampling',
	PENALTIES: 'penalties',
	AGENTIC: 'agentic',
	DEVELOPER: 'developer',
	TOOLS: 'tools',
	IMPORT_EXPORT: 'import-export'
} as const;

export const ROUTES = {
	/** Root — start of the app. */
	START: '#/',
	/** New chat — root with new chat query param. */
	NEW_CHAT: `?${URL_PARAMS.NEW_CHAT}=true#/`,
	/** Chat base — for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** Settings base — for dynamic settings URLs use RouterService. */
	SETTINGS: '#/settings',
	/** Search — mobile-only full-page conversation search. */
	SEARCH: '#/search'
} as const;
