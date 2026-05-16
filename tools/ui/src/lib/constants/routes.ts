export const NEW_CHAT_PARAM = 'new_chat';

/** Settings section slugs — used for routes and navigation. */
export const SETTINGS_SECTION_SLUGS = {
	GENERAL: 'general',
	DISPLAY: 'display',
	SAMPLING: 'sampling',
	PENALTIES: 'penalties',
	AGENTIC: 'agentic',
	DEVELOPER: 'developer',
	MCP: 'mcp',
	TOOLS: 'tools',
	IMPORT_EXPORT: 'import-export'
} as const;

export const ROUTES = {
	/** Root — start of the app. */
	START: '#/',
	/** New chat — root with new chat query param. */
	NEW_CHAT: `?${NEW_CHAT_PARAM}=true#/`,
	/** Chat base — for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** Settings base — for dynamic settings URLs use RouterService. */
	SETTINGS: '#/settings'
} as const;
