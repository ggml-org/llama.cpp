/**
 * recentMentionsStore - last-used `@`-mention entries, persisted to localStorage.
 *
 * The mention picker surfaces this list when the user opens it with no
 * characters typed after `@`, so they're one keystroke away from re-using
 * a file or folder they picked recently. The list is capped at 20 entries,
 * deduped by absolute path, and most-recent-first.
 *
 * The store conforms to `FileMentionEntry` so the picker can render it
 * with no shape conversion.
 *
 * Persistence notes:
 * - strict-shape validation on read: any item missing `path`, `name` or
 *   `type` is dropped rather than crashing the picker.
 * - storage failures are logged via `console.warn` only, matching the
 *   models store's style - we don't surface a toast for this; the
 *   recents list is a quality-of-life feature, not user data.
 */

import { browser } from '$app/environment';
import { RECENT_MENTIONS_LOCALSTORAGE_KEY } from '$lib/constants';
import type { FileMentionEntry } from '$lib/types';

const MAX_RECENT_MENTIONS = 20;

function isValidEntry(value: unknown): value is FileMentionEntry {
	if (!value || typeof value !== 'object') return false;
	const entry = value as Partial<FileMentionEntry>;
	return (
		typeof entry.path === 'string' &&
		entry.path.length > 0 &&
		typeof entry.name === 'string' &&
		entry.name.length > 0 &&
		(entry.type === 'file' || entry.type === 'directory')
	);
}

class RecentMentionsStore {
	items = $state<FileMentionEntry[]>(this.loadFromStorage());

	/**
	 * Push an entry to the front, dedupe by path, cap to MAX.
	 * Capitalizes the picker on the user's recent context.
	 *
	 * @param entry - The search entry the user just picked.
	 */
	add(entry: FileMentionEntry) {
		const withoutExisting = this.items.filter((e) => e.path !== entry.path);
		const next = [entry, ...withoutExisting].slice(0, MAX_RECENT_MENTIONS);
		this.items = next;
		this.persist(next);
	}

	private persist(next: FileMentionEntry[]) {
		if (!browser) return;
		try {
			localStorage.setItem(RECENT_MENTIONS_LOCALSTORAGE_KEY, JSON.stringify(next));
		} catch (err) {
			console.warn('[recentMentionsStore] Failed to persist:', err);
		}
	}

	private loadFromStorage(): FileMentionEntry[] {
		if (!browser) return [];
		try {
			const raw = localStorage.getItem(RECENT_MENTIONS_LOCALSTORAGE_KEY);
			if (!raw) return [];
			const parsed: unknown = JSON.parse(raw);
			if (!Array.isArray(parsed)) return [];
			// Validate each, drop malformed entries, cap to MAX.
			return parsed.filter(isValidEntry).slice(0, MAX_RECENT_MENTIONS);
		} catch {
			return [];
		}
	}
}

export const recentMentionsStore = new RecentMentionsStore();
