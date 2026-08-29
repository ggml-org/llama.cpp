import { browser } from '$app/environment';
import { NEW_CHAT_DRAFT_KEY, PROMPT_HISTORY_CHANGED_EVENT, SETTINGS_KEYS } from '$lib/constants';
import { conversationsStore, settingsStore } from '$lib/stores';
import {
	loadPromptHistoryBuckets,
	savePromptHistoryBuckets
} from '$lib/utils/prompt-history-storage';
import {
	getPromptHistoryEntries,
	pushPromptHistory,
	recallNext,
	recallPrevious,
	setPromptHistoryEntries,
	type PromptHistoryBuckets,
	type PromptHistoryCursor,
	type PromptHistoryScope
} from '$lib/utils/prompt-history';

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
	const initial = activeEntries(loadPromptHistoryBuckets());
	let entries = $state<string[]>(initial);
	let cursor = $state<PromptHistoryCursor>({ draft: '', index: initial.length });

	function reload() {
		const next = activeEntries(loadPromptHistoryBuckets());

		entries = next;
		cursor = { draft: '', index: next.length };
	}

	$effect(() => {
		void currentScope();
		void currentSessionId();
		reload();

		if (!browser) {
			return;
		}

		window.addEventListener(PROMPT_HISTORY_CHANGED_EVENT, reload);

		return () => {
			window.removeEventListener(PROMPT_HISTORY_CHANGED_EVENT, reload);
		};
	});

	function record(text: string) {
		const store = loadPromptHistoryBuckets();
		const scope = currentScope();
		const sessionId = currentSessionId();
		const updated = pushPromptHistory(getPromptHistoryEntries(store, scope, sessionId), text);

		savePromptHistoryBuckets(setPromptHistoryEntries(store, scope, sessionId, updated));
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
