export enum ColorMode {
	DARK = 'dark',
	LIGHT = 'light',
	SYSTEM = 'system'
}

/** The light surface theme: Paper (white panels on a faint grey page) or Matte (grey glass on white). */
export enum SurfaceTheme {
	MATTE = 'matte',
	PAPER = 'paper'
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
