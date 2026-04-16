import { Database, Settings, Search, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';
export const APP_NAME = 'llama.cpp';

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	activeRouteId?: string;
	activeRoutePrefix?: string;
}

export const DESKTOP_ICON_STRIP_ICONS: DesktopIconStripItem[] = [
	{ icon: SquarePen, tooltip: 'New Chat', route: '?new_chat=true#/' },
	{ icon: Search, tooltip: 'Search' },
	{
		icon: McpLogo,
		tooltip: 'MCP Servers',
		route: '#/settings/mcp',
		activeRouteId: '/settings/mcp'
	},
	{
		icon: Database,
		tooltip: 'Import / Export',
		route: '#/settings/import-export',
		activeRouteId: '/settings/import-export'
	},
	{
		icon: Settings,
		tooltip: 'Settings',
		route: '#/settings/chat/general',
		activeRoutePrefix: '/settings/chat'
	}
];
