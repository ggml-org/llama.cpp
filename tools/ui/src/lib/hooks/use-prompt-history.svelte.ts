import { browser } from '$app/environment';
import { PROMPT_HISTORY_LOCALSTORAGE_KEY } from '$lib/constants';
import {
	parsePromptHistory,
	pushPromptHistory,
	recallNext,
	recallPrevious,
	type PromptHistoryCursor
} from '$lib/utils/prompt-history';

function loadEntries(): string[] {
	if (!browser) {
		return [];
	}

	try {
		return parsePromptHistory(localStorage.getItem(PROMPT_HISTORY_LOCALSTORAGE_KEY));
	} catch {
		return [];
	}
}

function saveEntries(entries: string[]) {
	if (!browser) {
		return;
	}

	try {
		localStorage.setItem(PROMPT_HISTORY_LOCALSTORAGE_KEY, JSON.stringify(entries));
	} catch (error) {
		console.error('[prompt-history] Failed to persist prompt history:', error);
	}
}

/**
 * Session prompt stack for the chat input: ArrowUp / swipe-up recall
 * previously sent prompts; ArrowDown / swipe-down walk back toward the
 * live draft.
 */
export function usePromptHistory() {
	const initial = loadEntries();
	let entries = $state<string[]>(initial);
	let cursor = $state<PromptHistoryCursor>({ draft: '', index: initial.length });

	function record(text: string) {
		entries = pushPromptHistory(entries, text);
		cursor = { draft: '', index: entries.length };
		saveEntries(entries);
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
