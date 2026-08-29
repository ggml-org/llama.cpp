import { browser } from '$app/environment';
import { PROMPT_HISTORY_CHANGED_EVENT, PROMPT_HISTORY_LOCALSTORAGE_KEY } from '$lib/constants';
import {
	appendImportedPrompts,
	emptyPromptHistoryBuckets,
	parsePromptHistoryBuckets,
	type PromptHistoryBuckets
} from '$lib/utils/prompt-history';

export function loadPromptHistoryBuckets(): PromptHistoryBuckets {
	if (!browser) {
		return emptyPromptHistoryBuckets();
	}

	try {
		return parsePromptHistoryBuckets(localStorage.getItem(PROMPT_HISTORY_LOCALSTORAGE_KEY));
	} catch {
		return emptyPromptHistoryBuckets();
	}
}

export function savePromptHistoryBuckets(store: PromptHistoryBuckets) {
	if (!browser) {
		return;
	}

	try {
		localStorage.setItem(PROMPT_HISTORY_LOCALSTORAGE_KEY, JSON.stringify(store));
		window.dispatchEvent(new Event(PROMPT_HISTORY_CHANGED_EVENT));
	} catch (error) {
		console.error('[prompt-history] Failed to persist prompt history:', error);
	}
}

export function clearAllPromptHistory() {
	if (!browser) {
		return;
	}

	try {
		localStorage.removeItem(PROMPT_HISTORY_LOCALSTORAGE_KEY);
		window.dispatchEvent(new Event(PROMPT_HISTORY_CHANGED_EVENT));
	} catch (error) {
		console.error('[prompt-history] Failed to clear prompt history:', error);
	}
}

export function addImportedConversationPrompts(sessionId: string, texts: string[]) {
	if (texts.length === 0) {
		return;
	}

	savePromptHistoryBuckets(appendImportedPrompts(loadPromptHistoryBuckets(), sessionId, texts));
}
