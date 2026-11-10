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
	SIDEBAR_COLLAPSED_SECTIONS_LOCALSTORAGE_KEY,
	SIDEBAR_EXPANDED_LOCALSTORAGE_KEY,
	SIDEBAR_MAX_WIDTH,
	SIDEBAR_MIN_WIDTH,
	SIDEBAR_SECTION_ORDER_LOCALSTORAGE_KEY,
	SIDEBAR_WIDTH_LOCALSTORAGE_KEY
} from '$lib/constants';
import { deviceStore } from '$lib/stores/device.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { remToPx } from '$lib/utils/css';
import { SvelteSet } from 'svelte/reactivity';

/** Sidebar sections the user folded, read once on load. */
function loadCollapsedSidebarSections(): SvelteSet<string> {
	if (!browser) return new SvelteSet<string>();

	try {
		const raw = localStorage.getItem(SIDEBAR_COLLAPSED_SECTIONS_LOCALSTORAGE_KEY);
		const parsed = raw ? (JSON.parse(raw) as unknown) : null;

		return new SvelteSet(
			Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : []
		);
	} catch {
		return new SvelteSet<string>();
	}
}

/** Section order the user dragged into place, read once on load. */
function loadSidebarSectionOrder(): string[] {
	if (!browser) return [];

	try {
		const parsed = JSON.parse(localStorage.getItem(SIDEBAR_SECTION_ORDER_LOCALSTORAGE_KEY) ?? '[]');

		return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [];
	} catch {
		return [];
	}
}

class UiStore {
	/** Sidebar conversation sections the user folded, kept across reloads. */
	collapsedSidebarSections = loadCollapsedSidebarSections();
	/** Set when a flow outside the chat wants the composer focused once it can take it. */
	composerFocusRequested = $state(false);

	/** Repo the Discover dialog selects when it opens, set by a model row. */
	discoverModelFocus = $state<string | null>(null);

	/** Open state of the Discover dialog. */
	discoverModelsOpen = $state(false);
	/** Sidebar sections showing all their chats past the preview, for this session. */
	expandedSidebarSections = new SvelteSet<string>();

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
	/** Section ids in the order the user dragged them into, kept across reloads. */
	sidebarSectionOrder = $state<string[]>(loadSidebarSectionOrder());
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

	isSidebarSectionExpanded(id: string): boolean {
		return this.expandedSidebarSections.has(id);
	}

	isSidebarSectionOpen(id: string): boolean {
		return !this.collapsedSidebarSections.has(id);
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

	/**
	 * Puts sections in the order the user gave them. One never moved, such as a
	 * section that was empty until now, keeps its default place: right before the next section in
	 * the default order that was.
	 */
	orderSidebarSections<T extends { id: string }>(sections: T[]): T[] {
		const keys: number[] = [];

		// past every known rank, so trailing unplaced sections stay last
		let following = this.sidebarSectionOrder.length;

		for (let index = sections.length - 1; index >= 0; index--) {
			const known = this.sidebarSectionOrder.indexOf(sections[index].id);

			if (known === -1) {
				keys[index] = following - 0.5;
			} else {
				following = known;
				keys[index] = known;
			}
		}

		// a stable sort keeps unplaced neighbours in their default order
		return sections
			.map((section, index) => ({ key: keys[index], section }))
			.sort((a, b) => a.key - b.key)
			.map(({ section }) => section);
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

	/**
	 * Remembers the order the user dragged the sections into.
	 *
	 * @param ids - The sections as the sidebar shows them now
	 */
	setSidebarSectionOrder(ids: string[]): void {
		// sections not shown right now (an empty Pinned) keep their place
		this.sidebarSectionOrder = [
			...ids,
			...this.sidebarSectionOrder.filter((id) => !ids.includes(id))
		];

		try {
			localStorage.setItem(
				SIDEBAR_SECTION_ORDER_LOCALSTORAGE_KEY,
				JSON.stringify(this.sidebarSectionOrder)
			);
		} catch {
			// the order still applies for this session
		}
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

	/** Fold or unfold a sidebar conversation section and remember it. */
	toggleSidebarSection(id: string): void {
		if (this.collapsedSidebarSections.has(id)) {
			this.collapsedSidebarSections.delete(id);
		} else {
			this.collapsedSidebarSections.add(id);
		}

		try {
			localStorage.setItem(
				SIDEBAR_COLLAPSED_SECTIONS_LOCALSTORAGE_KEY,
				JSON.stringify([...this.collapsedSidebarSections])
			);
		} catch {
			// the fold still applies for this session
		}
	}

	/** Show all of a sidebar section's chats, or only its first ones again. */
	toggleSidebarSectionExpanded(id: string): void {
		if (this.expandedSidebarSections.has(id)) {
			this.expandedSidebarSections.delete(id);
		} else {
			this.expandedSidebarSections.add(id);
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
