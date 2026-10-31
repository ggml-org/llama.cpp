/**
 * uiStore - Shared UI/layout state
 *
 * Holds cross-component UI state that does not belong to a single component
 * (e.g. the desktop sidebar's expanded/collapsed state, which the sidebar
 * controls and the chat tab bar reacts to).
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { page } from '$app/state';
import {
	MODELS_ROUTE_ID,
	ROUTES,
	SETTINGS_KEYS,
	SIDEBAR_EXPANDED_LOCALSTORAGE_KEY,
	SIDEBAR_MAX_WIDTH,
	SIDEBAR_MIN_WIDTH,
	SIDEBAR_WIDTH_LOCALSTORAGE_KEY
} from '$lib/constants';
import { deviceStore } from '$lib/stores/device.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { remToPx } from '$lib/utils/css';

class UiStore {
	/** Set when a flow outside the chat wants the composer focused once it can take it. */
	composerFocusRequested = $state(false);

	/** Repo the Discover dialog selects when it opens, set by a model row. */
	discoverModelFocus = $state<string | null>(null);

	/** Open state of the Discover dialog. */
	discoverModelsOpen = $state(false);
	/** Whether the main chat form is expanded into its large editor. */
	isComposerExpanded = $state(false);

	/** Set while the user drags the sidebar edge, so width transitions can step aside. */
	isSidebarResizing = $state(false);

	/** Model the details drawer shows, a qualified id or a raw model name. */
	modelDetailsFocus = $state<string | null>(null);
	/** Route the models page returns to: the last page visited before it. */
	modelsReturnRoute: string = ROUTES.START;
	/** Bumped to close every open models selector; a counter, so no selector has to reset it. */
	modelsSelectorCloseSignal = $state(0);
	/** Expanded desktop sidebar width in px; null keeps the stylesheet default. */
	sidebarWidth = $state<number | null>(this.loadSidebarWidth());

	#isSidebarExpanded = $state(this.loadSidebarExpanded());

	/** Whether Discover is turned on; every Discover entry point hides without it. */
	get isDiscoverEnabled(): boolean {
		return Boolean(settingsStore.getConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS));
	}

	/** Whether the model details drawer is open. */
	get isModelDetailsOpen(): boolean {
		return this.modelDetailsFocus !== null;
	}

	/** Whether the models manager route is the current page. */
	get isModelsPageOpen(): boolean {
		return page.route.id === MODELS_ROUTE_ID;
	}

	/** Whether the desktop sidebar is expanded (open); remembered across reloads on desktop. */
	get isSidebarExpanded(): boolean {
		return this.#isSidebarExpanded;
	}

	set isSidebarExpanded(expanded: boolean) {
		this.#isSidebarExpanded = expanded;

		// a phone opens the sidebar as an overlay, which is no layout to restore
		if (!browser || deviceStore.isMobile) return;

		localStorage.setItem(SIDEBAR_EXPANDED_LOCALSTORAGE_KEY, String(expanded));
	}

	/** Close the model details drawer. */
	closeModelDetails(): void {
		this.modelDetailsFocus = null;
	}

	/** Leave the models page for the route it was opened from. */
	async closeModelsManager(): Promise<void> {
		if (!this.isModelsPageOpen) return;

		await goto(this.modelsReturnRoute);
	}

	/** Close any open models selector. */
	closeModelsSelector(): void {
		this.modelsSelectorCloseSignal++;
	}

	/** Open the model details drawer over the current page, without leaving it. */
	openModelDetails(focus: string): void {
		this.modelDetailsFocus = focus;
	}

	/** Open the Discover dialog, focused on one repo; a no-op while Discover is off. */
	openModelsDiscover(focus?: string): void {
		if (!this.isDiscoverEnabled) return;

		this.discoverModelFocus = focus ?? null;
		this.discoverModelsOpen = true;
	}

	/** Navigate to the models manager. */
	openModelsManager(): void {
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

	/** Set the expanded sidebar width in px, or null to go back to the default. */
	setSidebarWidth(width: number | null): void {
		this.sidebarWidth = width;

		if (!browser) return;

		if (width === null) {
			localStorage.removeItem(SIDEBAR_WIDTH_LOCALSTORAGE_KEY);
		} else {
			localStorage.setItem(SIDEBAR_WIDTH_LOCALSTORAGE_KEY, String(width));
		}
	}

	private loadSidebarExpanded(): boolean {
		if (!browser || deviceStore.isMobile) return false;

		return localStorage.getItem(SIDEBAR_EXPANDED_LOCALSTORAGE_KEY) === 'true';
	}

	private loadSidebarWidth(): number | null {
		if (!browser) return null;

		const width = Number(localStorage.getItem(SIDEBAR_WIDTH_LOCALSTORAGE_KEY));

		if (!Number.isFinite(width) || width <= 0) return null;

		// a width saved under older bounds is pulled back inside the current ones
		const clamped = Math.min(
			remToPx(SIDEBAR_MAX_WIDTH),
			Math.max(remToPx(SIDEBAR_MIN_WIDTH), width)
		);

		return Math.round(clamped);
	}
}

export const uiStore = new UiStore();
