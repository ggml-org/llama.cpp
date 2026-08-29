import { browser } from '$app/environment';
import { NEW_CHAT_DRAFT_KEY, PROMPT_HISTORY_LOCALSTORAGE_KEY, SETTINGS_KEYS } from '$lib/constants';
import { conversationsStore, settingsStore } from '$lib/stores';
import {
	getPromptHistoryEntries,
	parsePromptHistoryBuckets,
	pushPromptHistory,
	recallNext,
	recallPrevious,
	setPromptHistoryEntries,
	type PromptHistoryBuckets,
	type PromptHistoryCursor,
	type PromptHistoryScope
} from '$lib/utils/prompt-history';

function loadBuckets(): PromptHistoryBuckets {
	if (!browser) {
		return { combined: [], sessions: {} };
	}

	try {
		return parsePromptHistoryBuckets(localStorage.getItem(PROMPT_HISTORY_LOCALSTORAGE_KEY));
	} catch {
		return { combined: [], sessions: {} };
	}
}

function saveBuckets(store: PromptHistoryBuckets) {
	if (!browser) {
		return;
	}

	try {
		localStorage.setItem(PROMPT_HISTORY_LOCALSTORAGE_KEY, JSON.stringify(store));
	} catch (error) {
		console.error('[prompt-history] Failed to persist prompt history:', error);
	}
}

function currentScope(): PromptHistoryScope {
	return settingsStore.config[SETTINGS_KEYS.PROMPT_HISTORY_PER_SESSION] === true
		? 'separate'
		: 'combine';
}

function currentSessionId(): string {
	return conversationsStore.activeConversation?.id ?? NEW_CHAT_DRAFT_KEY;
}

function activeEntries(store: PromptHistoryBuckets): string[] {
	return getPromptHistoryEntries(store, currentScope(), currentSessionId());
}

/**
 * Session prompt stack for the chat input: ArrowUp / swipe-up recall
 * previously sent prompts; ArrowDown / swipe-down walk back toward the
 * live draft. Combined and per-session lists are stored separately so
 * the user can switch scope without losing either list.
 */
export function usePromptHistory() {
	const initial = activeEntries(loadBuckets());
	let entries = $state<string[]>(initial);
	let cursor = $state<PromptHistoryCursor>({ draft: '', index: initial.length });

	$effect(() => {
		const next = activeEntries(loadBuckets());

		entries = next;
		cursor = { draft: '', index: next.length };
	});

	function record(text: string) {
		const store = loadBuckets();
		const scope = currentScope();
		const sessionId = currentSessionId();
		const updated = pushPromptHistory(getPromptHistoryEntries(store, scope, sessionId), text);
		saveBuckets(setPromptHistoryEntries(store, scope, sessionId, updated));
		entries = updated;
		cursor = { draft: '', index: updated.length };
	}

	function previous(current: string): string | null {
		const recalled = recallPrevious(entries, cursor, current);

		if (!recalled) {
			return null;
		}

		cursor = recalled.cursor;

		return recalled.value;
	}

	function next(current: string): string | null {
		const recalled = recallNext(entries, cursor, current);

		if (!recalled) {
			return null;
		}

		cursor = recalled.cursor;

		return recalled.value;
	}

	return { next, previous, record };
}
