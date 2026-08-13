import { ROUTES } from './routes.constants';
import { Package, Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import { ToolSource } from '$lib/enums/tools.enums';
import type { DesktopIconStripItem } from '$lib/types';

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

/** Data attributes for app-level DOM contracts. */
export const UI_DATA_ATTRS = {
	ACTIVE: 'data-active',
	CONVERSATION_ROW: 'data-conversation-row',
	HIGHLIGHT_THEME_PREVIEW: 'data-highlight-theme-preview',
	PICKER_INDEX: 'data-picker-index',
	RESULT_INDEX: 'data-result-index',
	THUMBNAIL_INDEX: 'data-thumbnail-index'
} as const;

export const TOOL_GROUP_LABELS = {
	[ToolSource.BUILTIN]: 'Built-in',
	[ToolSource.CUSTOM]: 'JSON Schema',
	[ToolSource.FRONTEND]: 'Browser'
} as const;

export const TOOL_SERVER_LABELS = {
	[ToolSource.BUILTIN]: 'Built-in Tools',
	[ToolSource.CUSTOM]: 'Custom Tools',
	[ToolSource.FRONTEND]: 'Browser Tools'
} as const;

export const TOOLTIP_DELAY_DURATION = 500;

export const VIEWPORT_GUTTER = 8;
export const MENU_OFFSET = 6;

export const PROCESSING_INFO_TIMEOUT = 2000;

/**
 * Statistics units labels
 */
export const STATS_UNITS = {
	TOKENS_PER_SECOND: 't/s'
} as const;

export const DEFAULT_MOBILE_BREAKPOINT = 768;

/** Icon used for the model selector and the `/model` slash command. */
export const MODEL_SELECTOR_ICON = Package;

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const MAX_HEIGHT_CODE_BLOCK = '22rem';

export const SIDEBAR_ACTIONS_ITEMS: DesktopIconStripItem[] = [
	{ icon: SquarePen, keys: ['shift', 'cmd', 'o'], route: ROUTES.NEW_CHAT, tooltip: 'New chat' },
	{ icon: Search, keys: ['cmd', 'k'], tooltip: 'Search' },
	{
		activeRouteId: '/mcp-servers',
		icon: McpLogo,
		route: ROUTES.MCP_SERVERS,
		tooltip: 'MCP Servers'
	},
	{
		activeUrlIncludes: '#/settings',
		icon: Settings,
		route: `${ROUTES.SETTINGS}/general`,
		tooltip: 'Settings'
	}
];
