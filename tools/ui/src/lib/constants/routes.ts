/** Query params the chat routes read from the URL. */
export const URL_PARAMS = {
	/** Load the selected model instead of waiting for the first message. */
	LOAD: 'load',
	/** Model to select. */
	MODEL: 'model',
	/** Start a new chat. */
	NEW_CHAT: 'new_chat',
	/** Prompt to send on arrival. */
	QUERY: 'q'
} as const;

/** Settings section slugs — used for routes and navigation. */
export const SETTINGS_SECTION_SLUGS = {
	AGENTIC: 'agentic',
	DEVELOPER: 'developer',
	DISPLAY: 'display',
	GENERAL: 'general',
	IMPORT_EXPORT: 'import-export',
	PENALTIES: 'penalties',
	SAMPLING: 'sampling',
	TOOLS: 'tools'
} as const;

export const ROUTES = {
	/** Chat base — for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** New chat — root with new chat query param. */
	NEW_CHAT: `?${URL_PARAMS.NEW_CHAT}=true#/`,
	/** Search — mobile-only full-page conversation search. */
	SEARCH: '#/search',
	/** Settings base — for dynamic settings URLs use RouterService. */
	SETTINGS: '#/settings',
	/** Root — start of the app. */
	START: '#/'
} as const;
