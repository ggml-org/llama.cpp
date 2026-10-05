/**
 * uiStore - Shared UI/layout state
 *
 * Holds cross-component UI state that does not belong to a single component
 * (e.g. the desktop sidebar's expanded/collapsed state, which the sidebar
 * controls and the chat tab bar reacts to).
 */

class UiStore {
	/** Whether the desktop sidebar is expanded (open). */
	isSidebarExpanded = $state(false);
}

export const uiStore = new UiStore();
