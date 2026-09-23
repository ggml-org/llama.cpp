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
	/** Model the manager reveals when it opens, a qualified id or a raw model name. */
	manageModelFocus = $state<string | null>(null);
	/** Open state of the models manager, driven from the sidebar and from model rows. */
	manageModelsOpen = $state(false);

	/** Open the models manager, optionally focused on one model. */
	openModelsManager(focus?: string): void {
		this.manageModelFocus = focus ?? null;
		this.manageModelsOpen = true;
	}
}

export const uiStore = new UiStore();
