import { MODEL_ICON } from './icons.constants';
import {
	SHORTCUT_MODELS,
	SHORTCUT_NEW_CHAT,
	SHORTCUT_SEARCH,
	SHORTCUT_SETTINGS
} from './keyboard-shortcuts.constants';
import { MODELS_ROUTE_ID } from './routes.constants';
import { Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import { SidebarAction, ToolSource } from '$lib/enums';
import type { DesktopIconStripItem } from '$lib/types';

/** Delay before a copied affordance resets from its check to the copy icon. */
export const COPY_FEEDBACK_RESET_MS = 1500;

/** How long a collapsible block's body takes to expand or collapse. */
export const COLLAPSIBLE_TRANSITION_MS = 200;

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

/** Data attributes for app-level DOM contracts. */
export const UI_DATA_ATTRS = {
	ACTIVE: 'data-active',
	ACTIVE_TAB: 'data-active-tab',
	CONVERSATION_ROW: 'data-conversation-row',
	HIGHLIGHT_THEME_PREVIEW: 'data-highlight-theme-preview',
	PICKER_INDEX: 'data-picker-index',
	RESULT_INDEX: 'data-result-index',
	THUMBNAIL_INDEX: 'data-thumbnail-index'
} as const;

export const TOOL_GROUP_LABELS = {
	[ToolSource.BROWSER]: 'Browser',
	[ToolSource.CUSTOM]: 'JSON Schema',
	[ToolSource.SERVER]: 'Server'
} as const;

export const TOOL_SERVER_LABELS = {
	[ToolSource.BROWSER]: 'Browser Tools',
	[ToolSource.CUSTOM]: 'Custom Tools',
	[ToolSource.SERVER]: 'Server Tools'
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

/** Models listed per remote provider before the "+ X more" line; search covers the rest. */
export const REMOTE_PROVIDER_MODEL_LIMIT = 12;

/** Orgs whose avatar is dark and needs inverting in dark mode. */
export const DARK_INVERT_AVATAR_ORGS = ['openai'];

/** Model rows mounted before the list scrolls; a local catalog can hold hundreds. */
export const MODEL_ROW_WINDOW = 50;

/** Models of one family shown before the rest fold behind a "show more" row. */
export const FAMILY_ROW_WINDOW = 6;

/** Recently used models kept per browser, most recent first. */
export const RECENT_MODEL_LIMIT = 20;

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

/** Collapsed desktop sidebar strip width, matching its `md:w-12`. */
export const SIDEBAR_COLLAPSED_WIDTH = '3rem';
/** Bounds the expanded desktop sidebar can be dragged between. */
export const SIDEBAR_MIN_WIDTH = '18rem';
export const SIDEBAR_MAX_WIDTH = '32rem';
/** Dragging the sidebar edge narrower than this snaps it to the pointer; letting go hides it. */
export const SIDEBAR_COLLAPSE_WIDTH = '9rem';
/** How far past its min or max width a drag can stretch the sidebar before it resists fully. */
export const SIDEBAR_OVERSTRETCH_PX = 80;
/** Spring-back duration when a drag ends outside the width bounds. */
export const SIDEBAR_SETTLE_MS = 300;
/** Duration of the sidebar's open and close slide, matching its `duration-200` transition. */
export const SIDEBAR_TOGGLE_MS = 200;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const MAX_HEIGHT_CODE_BLOCK = '22rem';

export const SIDEBAR_ACTIONS_ITEMS: DesktopIconStripItem[] = [
	{
		action: SidebarAction.NEW_CHAT,
		icon: SquarePen,
		keys: SHORTCUT_NEW_CHAT,
		tooltip: 'New chat'
	},
	{ action: SidebarAction.SEARCH, icon: Search, keys: SHORTCUT_SEARCH, tooltip: 'Search' },
	{
		action: SidebarAction.MANAGE_MODELS,
		activeRouteId: MODELS_ROUTE_ID,
		icon: MODEL_ICON,
		keys: SHORTCUT_MODELS,
		tooltip: 'Models'
	},
	{
		action: SidebarAction.MCP,
		icon: McpLogo,
		tooltip: 'MCP'
	},
	{
		action: SidebarAction.SETTINGS,
		icon: Settings,
		keys: SHORTCUT_SETTINGS,
		tooltip: 'Settings'
	}
];
