import { Settings, Search, SquarePen, ScanText } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';
import { ROUTES } from './routes';

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

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
	{ icon: SquarePen, tooltip: 'New chat', route: ROUTES.NEW_CHAT, keys: ['shift', 'cmd', 'o'] },
	{ icon: Search, tooltip: 'Search', keys: ['cmd', 'k'] },
	{
		icon: McpLogo,
		tooltip: 'MCP Servers',
		route: ROUTES.MCP_SERVERS,
		activeRouteId: '/(manage)/mcp-servers'
	},
	{
		icon: ScanText,
		tooltip: 'Prompts',
		route: ROUTES.PROMPTS,
		activeRouteId: '/(manage)/prompts'
	},
	{
		icon: Settings,
		tooltip: 'Settings',
		route: `${ROUTES.SETTINGS}/general`,
		activeUrlIncludes: '#/settings'
	}
];

export const CHAT_FORM_PLACEHOLDER_GREETING = 'Hello there!';

export const CHAT_FORM_PLACEHOLDER = {
  DEFAULT: 'Type a message',
  VOICE_MODE_AVAILABLE: 'Type a message or use your voice',
  PROMPTS: 'Use / to attach a prompt',
  RESOURCES: 'Use @ to attach resources'
}
