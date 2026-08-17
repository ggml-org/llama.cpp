/**
 * tabsStore - Reactive State Store for Browser-Style Conversation Tabs
 *
 * Tracks which conversations and new-chat screens are open as tabs in the
 * chat layout, in order. Every tab - real conversation or unsaved new-chat
 * tab - is a `#/chat/<id>` route; new-chat tabs simply carry a temporary
 * conversation id that is only persisted to the database once a message is
 * sent (see conversationsStore.temporaryConversations).
 *
 * **Architecture & Relationships:**
 * - **conversationsStore**: owns conversation data; calls `removeTabs()` /
 *   `close()` when conversations are deleted. This store never imports it,
 *   so there is no circular dependency - tab names are resolved by the
 *   ChatTabs component from conversationsStore.
 * - Tab order persists to localStorage and is pruned against the loaded
 *   conversation list on init. Unsaved new-chat tabs are dropped on reload
 *   (they are not in the database), which matches browser behavior.
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { CHAT_TABS_LOCALSTORAGE_KEY, ROUTES } from '$lib/constants';
import { RouterService } from '$lib/services/router.service';
import { untrack } from 'svelte';

class TabsStore {
	/** Ordered tab ids: conversation ids and temporary new-chat ids */
	openTabs = $state<string[]>([]);

	/** False until init() has read the persisted tabs; save() is a no-op before that */
	private initialized = false;

	/** Navigate to a tab */
	async activate(id: string): Promise<void> {
		await goto(RouterService.chat(id));
	}

	/** Remove all tabs (e.g. after deleting all conversations) */
	clear(): void {
		this.openTabs = [];
		this.save();
	}

	/**
	 * Close a tab. When it belongs to the active route, navigate to the left
	 * neighbor (or the right one when the closed tab was leftmost), falling
	 * back to the new-chat screen when no tabs remain.
	 * @param id - Tab id to close
	 * @param activeTabId - Tab id of the current route, if any
	 */
	async close(id: string, activeTabId: string | null): Promise<void> {
		const idx = this.openTabs.indexOf(id);

		if (idx === -1) return;

		this.openTabs = this.openTabs.filter((tabId) => tabId !== id);
		this.save();

		if (id !== activeTabId) return;

		const target = (idx > 0 ? this.openTabs[idx - 1] : this.openTabs[0]) ?? null;

		if (target) {
			await goto(RouterService.chat(target));
		} else {
			await goto(ROUTES.START);
		}
	}

	/**
	 * Load persisted tabs, dropping conversation ids that no longer exist.
	 * Called once from initStores() after conversations are loaded.
	 * Merges with (rather than replaces) current openTabs: the chat layout
	 * syncs the route's tab before this async init completes, and replacing
	 * here would drop it.
	 * @param validIds - Ids of conversations present in the database
	 */
	init(validIds: string[]): void {
		if (!browser) return;

		const valid = new Set(validIds);
		const persisted = this.load().filter((id) => valid.has(id));
		const extras = this.openTabs.filter((id) => valid.has(id) && !persisted.includes(id));

		this.openTabs = [...persisted, ...extras];
		this.initialized = true;
		this.save();
	}

	/**
	 * Remove tabs without navigating. Used when conversations are deleted
	 * while some other conversation stays open.
	 * @param ids - Tab ids to drop
	 */
	removeTabs(ids: string[]): void {
		const removed = new Set(ids);
		const next = this.openTabs.filter((id) => !removed.has(id));

		if (next.length !== this.openTabs.length) {
			this.openTabs = next;
			this.save();
		}
	}

	/**
	 * Sync the tab strip with the route. Called from the chat layout on every
	 * navigation, so any way of reaching a conversation or new-chat tab opens
	 * a tab for it.
	 * @param id - The conversation (or temporary new-chat) id of the route
	 */
	syncWithRoute(id: string): void {
		// untrack: callers invoke this from an effect keyed on the route, and
		// reading openTabs here would subscribe that effect to openTabs too -
		// closing the active tab would then re-run the effect and re-add the tab
		untrack(() => {
			if (!this.openTabs.includes(id)) {
				this.openTabs = [...this.openTabs, id];
				this.save();
			}
		});
	}

	private load(): string[] {
		try {
			const raw = localStorage.getItem(CHAT_TABS_LOCALSTORAGE_KEY);
			const parsed: unknown = raw ? JSON.parse(raw) : [];

			return Array.isArray(parsed) ? parsed.filter((id) => typeof id === 'string') : [];
		} catch {
			return [];
		}
	}

	private save(): void {
		// never write before init has read the persisted tabs, or an early
		// route sync (layout effect runs before async init) would clobber them
		if (!browser || !this.initialized) return;

		localStorage.setItem(CHAT_TABS_LOCALSTORAGE_KEY, JSON.stringify(this.openTabs));
	}
}

export const tabsStore = new TabsStore();
