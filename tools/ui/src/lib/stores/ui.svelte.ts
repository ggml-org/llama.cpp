/**
 * uiStore - Shared UI/layout state
 *
 * Holds cross-component UI state that does not belong to a single component
 * (e.g. the desktop sidebar's expanded/collapsed state, which the sidebar
 * controls and the chat tab bar reacts to).
 */

class UiStore {
	/** Set when a flow outside the chat wants the composer focused once it can take it. */
	composerFocusRequested = $state(false);

	/** Repo the Discover view selects when it opens, set by a model row. */
	discoverModelFocus = $state<string | null>(null);
	discoverModelsOpen = $state(false);
	/** Whether the desktop sidebar is expanded (open). */
	isSidebarExpanded = $state(false);
	/** Model the manager reveals when it opens, a qualified id or a raw model name. */
	manageModelFocus = $state<string | null>(null);
	/** Open state of the models manager, driven from the sidebar and from model rows. */
	manageModelsOpen = $state(false);

	openDiscoverModels(): void {
		this.discoverModelFocus = null;
		this.discoverModelsOpen = true;
	}

	/** Open the models dialog on Discover, focused on one repo. */
	openModelsDiscover(focus?: string): void {
		this.discoverModelFocus = focus ?? null;
		this.discoverModelsOpen = true;
	}

	/** Open the models manager, optionally focused on one model. */
	openModelsManager(focus?: string): void {
		this.manageModelFocus = focus ?? null;
		this.manageModelsOpen = true;
	}

	/** Ask the composer to take focus, e.g. after a dialog closes onto the chat. */
	requestComposerFocus(): void {
		this.composerFocusRequested = true;
	}
}

export const uiStore = new UiStore();
