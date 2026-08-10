import { ROUTES } from './routes';
import { Package, Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

/** Icon used for the model selector and the `/model` slash command. */
export const MODEL_SELECTOR_ICON = Package;

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const MAX_HEIGHT_CODE_BLOCK = '22rem';

export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}

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
