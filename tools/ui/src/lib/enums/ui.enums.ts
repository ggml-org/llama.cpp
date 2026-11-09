export enum ColorMode {
	DARK = 'dark',
	LIGHT = 'light',
	SYSTEM = 'system'
}

export enum TooltipSide {
	BOTTOM = 'bottom',
	LEFT = 'left',
	RIGHT = 'right',
	TOP = 'top'
}

/**
 * ScrollCarousel arrow placement.
 */
export enum ScrollCarouselVariant {
	CENTER = 'center',
	TOP = 'top'
}

/**
 * Sidebar icon strip actions handled directly by the sidebar.
 */
export enum SidebarAction {
	MANAGE_MODELS = 'manage-models',
	MCP = 'mcp',
	NEW_CHAT = 'new-chat',
	SEARCH = 'search',
	SETTINGS = 'settings'
}

/** Kind of a sidebar conversation section, which can be dragged into another order. */
export enum SidebarSectionKind {
	PINNED = 'pinned',
	RECENT = 'recent'
}

/**
 * MCP prompt display variant
 */
export enum McpPromptVariant {
	ATTACHMENT = 'attachment',
	MESSAGE = 'message'
}

/**
 * URL prefixes for protocol detection
 */
export enum UrlProtocol {
	DATA = 'data:',
	FILE = 'file:',
	HTTP = 'http:',
	HTTPS = 'https:',
	WEBSOCKET = 'ws:',
	WEBSOCKET_SECURE = 'wss:'
}

export enum HtmlInputType {
	FILE = 'file'
}

/**
 * Alert level that drives the context gauge dial color.
 */
export enum ColorLevel {
	CRITICAL = 'critical',
	NEUTRAL = 'neutral',
	OK = 'ok',
	WARNING = 'warning'
}

/**
 * Which menu an item's actions render in.
 */
export enum MenuKind {
	CONTEXT = 'context',
	DROPDOWN = 'dropdown'
}
