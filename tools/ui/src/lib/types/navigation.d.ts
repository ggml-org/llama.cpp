import type { MenuKind, SidebarAction } from '$lib/enums';
import type { Component, Snippet } from 'svelte';

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
	/**
	 * Flyout rendered in place of the row in a desktop menu, through the kit of
	 * the menu it is in; a phone's menus keep the row and its `onclick`.
	 */
	submenu?: Snippet<[{ close: () => void; kind: MenuKind }]>;
	variant?: 'default' | 'destructive';
	disabled?: boolean;
	/** Shown on hover, e.g. why a disabled action is not available. */
	tooltip?: string;
	shortcut?: string[];
	/** Draw a separator above this action. */
	separator?: boolean;
}
