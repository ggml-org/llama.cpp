import type { SidebarAction } from '$lib/enums';
import type { Component } from 'svelte';

/**
 * A single clickable action in the desktop sidebar icon strip.
 */
export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	/** Custom action handled by the sidebar, e.g. opening a new-chat tab */
	action?: SidebarAction;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}

/**
 * One action in an item's menu, rendered the same way in its "more actions"
 * dropdown and its right-click context menu.
 */
export interface MenuAction {
	icon: Component;
	label: string;
	onclick: (event: Event) => void;
	variant?: 'default' | 'destructive';
	disabled?: boolean;
	/** Shown on hover, e.g. why a disabled action is not available. */
	tooltip?: string;
	shortcut?: string[];
	/** Draw a separator above this action. */
	separator?: boolean;
}
