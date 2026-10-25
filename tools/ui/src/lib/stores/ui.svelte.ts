/**
 * uiStore - Shared UI/layout state
 *
 * Holds cross-component UI state that does not belong to a single component
 * (e.g. the desktop sidebar's expanded/collapsed state, which the sidebar
 * controls and the chat tab bar reacts to).
 */

import { goto } from '$app/navigation';
import { page } from '$app/state';
import { MODELS_ROUTE_ID, ROUTES, SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ModelOption } from '$lib/types/models';

class UiStore {
	/** Set when a flow outside the chat wants the composer focused once it can take it. */
	composerFocusRequested = $state(false);

	/** Repo the Discover dialog selects when it opens, set by a model row. */
	discoverModelFocus = $state<string | null>(null);
	/** Open state of the Discover dialog. */
	discoverModelsOpen = $state(false);

	/** Whether the desktop sidebar is expanded (open). */
	isSidebarExpanded = $state(false);
	/** Model the manager reveals when it opens, a qualified id or a raw model name. */
	manageModelFocus = $state<string | null>(null);
	/** Model the information dialog shows; a phone opens it instead of the manager. */
	modelInformation = $state<ModelOption | null>(null);
	/** Route the models page returns to: the last page visited before it. */
	modelsReturnRoute: string = ROUTES.START;

	/** Whether Discover is turned on; every Discover entry point hides without it. */
	get isDiscoverEnabled(): boolean {
		return Boolean(settingsStore.getConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS));
	}

	/** Whether the models manager route is the current page. */
	get isModelsPageOpen(): boolean {
		return page.route.id === MODELS_ROUTE_ID;
	}

	/** Leave the models page for the route it was opened from. */
	async closeModelsManager(): Promise<void> {
		if (!this.isModelsPageOpen) return;

		await goto(this.modelsReturnRoute);
	}

	/** Open the model information dialog, a phone's stand-in for the manager pane. */
	openModelInformation(option: ModelOption): void {
		this.modelInformation = option;
	}

	/** Open the Discover dialog, focused on one repo; a no-op while Discover is off. */
	openModelsDiscover(focus?: string): void {
		if (!this.isDiscoverEnabled) return;

		this.discoverModelFocus = focus ?? null;
		this.discoverModelsOpen = true;
	}

	/** Navigate to the models manager, optionally focused on one model. */
	openModelsManager(focus?: string): void {
		this.manageModelFocus = focus ?? null;

		if (this.isModelsPageOpen) return;

		void goto(ROUTES.MODELS);
	}

	/** Remember the current route as the models page's way back, unless it is that page. */
	rememberModelsReturnRoute(): void {
		if (this.isModelsPageOpen) return;

		this.modelsReturnRoute = page.url.hash || ROUTES.START;
	}

	/** Ask the composer to take focus, e.g. after a dialog closes onto the chat. */
	requestComposerFocus(): void {
		this.composerFocusRequested = true;
	}
}

export const uiStore = new UiStore();
