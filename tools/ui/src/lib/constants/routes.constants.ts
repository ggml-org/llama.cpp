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
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** Search — mobile-only full-page conversation search. */
	SEARCH: '#/search',
	/** Settings base — for dynamic settings URLs use RouterService. */
	SETTINGS: '#/settings',
	/** Exit destination for the settings view (fallback when no referrer). */
	SETTINGS_EXIT: '#/',
	/** Root — start of the app. */
	START: '#/'
} as const;
