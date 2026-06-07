export enum ColorMode {
	LIGHT = 'light',
	DARK = 'dark',
	SYSTEM = 'system'
}

export enum TooltipSide {
	TOP = 'top',
	RIGHT = 'right',
	BOTTOM = 'bottom',
	LEFT = 'left'
}

/**
 * MCP prompt display variant
 */
export enum McpPromptVariant {
	MESSAGE = 'message',
	ATTACHMENT = 'attachment'
}

/**
 * URL prefixes for protocol detection
 */
export enum UrlProtocol {
	DATA = 'data:',
	HTTP = 'http:',
	HTTPS = 'https:',
	WEBSOCKET = 'ws:',
	WEBSOCKET_SECURE = 'wss:'
}

export enum HtmlInputType {
	FILE = 'file'
}
