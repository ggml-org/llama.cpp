/**
 * tabsStore - Reactive State Store for Browser-Style Conversation Tabs
 *
 * Tracks which conversations and new-chat screens are open as tabs in the
 * chat layout, in order. Conversation tabs hold the conversation id; the
 * new-chat tab of the bare `#/` route is the `NEW_CHAT_TAB_ID` sentinel, and
 * a project's new-chat screen has a sentinel of its own (see
 * `projectNewChatTabId`). The regular view and a project view show different
 * slices of the same tabs (see `resolveTabView`).
 *
 * Tabs open like in a browser: a plain navigation shows the new target in
 * the current tab, and only `openInNewTab()` adds a tab. A project home is
 * never a tab, so a chat opened from it gets a tab of its own.
 *
 * **Architecture & Relationships:**
 * - **conversationsStore**: owns conversation data; calls `removeTabs()` /
 *   `closeConversation()` when conversations are deleted. This store never
 *   imports it, so there is no circular dependency - tab names are resolved
 *   by the ChatTabs component from conversationsStore.
 * - Tab order persists to localStorage and is pruned against the loaded
 *   conversation list on init. The new-chat tab is kept across reloads.
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import {
	CONVERSATION_TABS_LOCALSTORAGE_KEY,
	NEW_CHAT_TAB_ID,
	PROJECT_HOME_TAB_ID,
	ROUTES
} from '$lib/constants';
import { RouterService } from '$lib/services/router.service';
import { projectOfNewChatTab, type TabView } from '$lib/utils/chat-tabs';
import { untrack } from 'svelte';

export class TabsStore {
	/** Ordered tab ids: conversation ids and the `NEW_CHAT_TAB_ID` sentinel */
	openTabs = $state<string[]>([]);

	/** Target of the last route the chat layout reported, the tab a plain navigation reuses */
	private current: string | null = null;

	/** False until init() has read the persisted tabs; save() is a no-op before that */
	private initialized = false;

	/** Route synced before init, applied once the persisted tabs are loaded */
	private pendingTarget: string | null = null;

	/** Navigate to a tab (the new-chat sentinel maps to the bare `#/` route) */
	async activate(id: string): Promise<void> {
		await goto(this.hrefFor(id));
	}

	/** Remove all tabs (e.g. after deleting all conversations) */
	clear(): void {
		this.openTabs = [];
		this.current = null;
		this.save();
	}

	/**
	 * Close a tab. When it belongs to the active route, navigate to the left
	 * neighbor (or the right one when the closed tab was leftmost) among the
	 * tabs the view shows, falling back to the view's start when none remain.
	 * @param id - Tab id to close
	 * @param isActive - Whether the tab is the current route
	 * @param view - Visible tabs and their links; all tabs when omitted
	 */
	async close(
		id: string,
		isActive: boolean,
		view: Pick<TabView, 'fallbackHref' | 'hrefFor' | 'ids'> = {
			fallbackHref: ROUTES.START,
			hrefFor: (tabId) => this.hrefFor(tabId),
			ids: this.openTabs
		}
	): Promise<void> {
		const idx = view.ids.indexOf(id);

		if (!this.openTabs.includes(id)) {
			// tab not tracked (e.g. Conversation tabs are off); still leave the
			// closed route for the view's start
			if (isActive) await goto(view.fallbackHref);

			return;
		}

		this.openTabs = this.openTabs.filter((tabId) => tabId !== id);
		this.save();

		if (!isActive) return;

		const visible = view.ids.filter((tabId) => tabId !== id);
		const next = (idx > 0 ? visible[idx - 1] : visible[0]) ?? null;

		await goto(next ? view.hrefFor(next) : view.fallbackHref);
	}

	/** Close the tab of the conversation open in the route, e.g. after deleting it */
	async closeConversation(id: string): Promise<void> {
		await this.close(id, true);
	}

	/**
	 * Close several tabs at once, e.g. the others or the ones to the right of
	 * a tab. When the active tab is among them, navigate to `keepId`, the tab
	 * the action was taken on.
	 * @param ids - Tab ids to close
	 * @param keepId - Tab that stays open
	 * @param activeId - Tab of the current route, if any
	 * @param view - Links of the tabs the view shows; all tabs when omitted
	 */
	async closeMany(
		ids: string[],
		keepId: string,
		activeId: string | null,
		view: Pick<TabView, 'hrefFor'> = { hrefFor: (tabId) => this.hrefFor(tabId) }
	): Promise<void> {
		this.removeTabs(ids);

		if (activeId && ids.includes(activeId)) await goto(view.hrefFor(keepId));
	}

	hrefFor(id: string): string {
		if (id === NEW_CHAT_TAB_ID) return ROUTES.START;

		const projectId = projectOfNewChatTab(id);

		return projectId ? RouterService.projectNewChat(projectId) : RouterService.chat(id);
	}

	/**
	 * Load persisted tabs, dropping conversation ids that no longer exist,
	 * then apply the route synced while loading.
	 * Called once from initStores() after conversations are loaded.
	 * @param validIds - Ids of conversations present in the database
	 */
	init(validIds: string[]): void {
		if (!browser) return;

		// new-chat sentinels are pseudo-tabs, not conversations, but they are
		// still kept so a reload on a new-chat screen does not drop its tab
		this.openTabs = this.load().filter(
			(id) => id === NEW_CHAT_TAB_ID || !!projectOfNewChatTab(id) || validIds.includes(id)
		);
		this.initialized = true;
		this.save();

		if (this.pendingTarget) {
			this.syncWithRoute(this.pendingTarget);
			this.pendingTarget = null;
		}
	}

	/**
	 * Add a tab at the end. Without `activate` the tab opens in the
	 * background, like a middle-click in a browser.
	 * @param id - What the new tab shows
	 * @param options.activate - Also navigate to the tab
	 */
	async openInNewTab(id: string, { activate = false } = {}): Promise<void> {
		if (!this.openTabs.includes(id)) {
			this.openTabs = [...this.openTabs, id];
			this.save();
		}

		if (activate) await this.activate(id);
	}

	/**
	 * Remove tabs without navigating. Used when conversations are deleted
	 * while some other conversation stays open.
	 * @param ids - Tab ids to drop
	 */
	removeTabs(ids: string[]): void {
		const next = this.openTabs.filter((id) => !ids.includes(id));

		if (next.length !== this.openTabs.length) {
			this.openTabs = next;
			this.save();
		}
	}

	/**
	 * Sync the tab strip with the route. Called from the chat layout on every
	 * navigation. A target already open activates its tab; otherwise the tab
	 * of the previous route now shows it, and only a navigation from a page
	 * without a tab adds one.
	 * @param target - Conversation id, `NEW_CHAT_TAB_ID` or `PROJECT_HOME_TAB_ID`
	 */
	syncWithRoute(target: string): void {
		// untrack: callers invoke this from an effect keyed on the route, and
		// reading openTabs here would subscribe that effect to openTabs too -
		// closing the active tab would then re-run the effect and re-add the tab
		untrack(() => {
			if (!this.initialized) {
				this.pendingTarget = target;

				return;
			}

			const previous = this.current;

			this.current = target;

			// the project home has no tab; leaving it opens one
			if (target === PROJECT_HOME_TAB_ID || this.openTabs.includes(target)) return;

			const idx = previous ? this.openTabs.indexOf(previous) : -1;

			this.openTabs =
				idx === -1
					? [...this.openTabs, target]
					: this.openTabs.map((id, i) => (i === idx ? target : id));
			this.save();
		});
	}

	private load(): string[] {
		try {
			const raw = localStorage.getItem(CONVERSATION_TABS_LOCALSTORAGE_KEY);
			const parsed: unknown = raw ? JSON.parse(raw) : [];

			if (Array.isArray(parsed)) return parsed.filter((id) => typeof id === 'string');

			// tabs saved while each project had its own strip: one strip per key
			if (parsed && typeof parsed === 'object') {
				return Object.values(parsed).flatMap((strip: { tabs?: unknown }) =>
					Array.isArray(strip?.tabs) ? strip.tabs.filter((id) => typeof id === 'string') : []
				);
			}

			return [];
		} catch {
			return [];
		}
	}

	private save(): void {
		// never write before init has read the persisted tabs, or an early
		// write would clobber them
		if (!browser || !this.initialized) return;

		localStorage.setItem(CONVERSATION_TABS_LOCALSTORAGE_KEY, JSON.stringify(this.openTabs));
	}
}

export const tabsStore = new TabsStore();
