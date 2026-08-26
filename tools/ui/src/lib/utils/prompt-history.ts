import { PROMPT_HISTORY_MAX_ENTRIES, PROMPT_HISTORY_SWIPE_MIN_PX } from '$lib/constants';

export type PromptHistorySwipe = 'up' | 'down';

export interface PromptHistoryCursor {
	/** Recalled index, or `entries.length` while editing the live draft. */
	index: number;
	draft: string;
}

/**
 * ArrowUp / swipe-up only when the caret is still on the first line,
 * so multiline editing keeps native line movement.
 */
export function isCaretOnFirstLine(value: string, caret: number): boolean {
	const offset = clampCaret(value, caret);

	return value.lastIndexOf('\n', Math.max(0, offset) - 1) === -1;
}

/**
 * ArrowDown / swipe-down only when the caret is on the last line.
 */
export function isCaretOnLastLine(value: string, caret: number): boolean {
	const offset = clampCaret(value, caret);

	return value.indexOf('\n', offset) === -1;
}

export function pushPromptHistory(
	entries: string[],
	next: string,
	max = PROMPT_HISTORY_MAX_ENTRIES
): string[] {
	const text = next.trim();

	if (!text) {
		return entries;
	}

	if (entries[entries.length - 1] === text) {
		return entries;
	}

	const pushed = [...entries, text];

	return pushed.length > max ? pushed.slice(pushed.length - max) : pushed;
}

export function parsePromptHistory(raw: string | null): string[] {
	if (!raw) {
		return [];
	}

	try {
		const parsed: unknown = JSON.parse(raw);

		if (!Array.isArray(parsed)) {
			return [];
		}

		return parsed.filter((item): item is string => typeof item === 'string' && item.trim() !== '');
	} catch {
		return [];
	}
}

export function recallPrevious(
	entries: string[],
	cursor: PromptHistoryCursor,
	current: string
): { cursor: PromptHistoryCursor; value: string } | null {
	if (entries.length === 0) {
		return null;
	}

	const index = normalizeIndex(entries, cursor.index);
	const draft = index === entries.length ? current : cursor.draft;

	if (index <= 0) {
		return null;
	}

	const nextIndex = index - 1;

	return {
		cursor: { draft, index: nextIndex },
		value: entries[nextIndex]
	};
}

export function recallNext(
	entries: string[],
	cursor: PromptHistoryCursor,
	_current: string
): { cursor: PromptHistoryCursor; value: string } | null {
	if (entries.length === 0) {
		return null;
	}

	const index = normalizeIndex(entries, cursor.index);

	if (index >= entries.length) {
		return null;
	}

	const nextIndex = index + 1;

	if (nextIndex >= entries.length) {
		return {
			cursor: { draft: cursor.draft, index: entries.length },
			value: cursor.draft
		};
	}

	return {
		cursor: { draft: cursor.draft, index: nextIndex },
		value: entries[nextIndex]
	};
}

export function swipeDirection(
	dx: number,
	dy: number,
	minPx = PROMPT_HISTORY_SWIPE_MIN_PX
): PromptHistorySwipe | null {
	if (Math.abs(dy) < minPx) {
		return null;
	}

	if (Math.abs(dy) <= Math.abs(dx) * 1.2) {
		return null;
	}

	return dy < 0 ? 'up' : 'down';
}

/** True when the element can still scroll in the swipe direction. */
export function canScrollInDirection(el: HTMLElement, direction: PromptHistorySwipe): boolean {
	const overflow = el.scrollHeight - el.clientHeight;

	if (overflow <= 1) {
		return false;
	}

	if (direction === 'up') {
		return el.scrollTop > 1;
	}

	return el.scrollTop + el.clientHeight < el.scrollHeight - 1;
}

function clampCaret(value: string, caret: number): number {
	if (!Number.isFinite(caret)) {
		return value.length;
	}

	return Math.max(0, Math.min(Math.floor(caret), value.length));
}

function normalizeIndex(entries: string[], index: number): number {
	if (!Number.isFinite(index)) {
		return entries.length;
	}

	return Math.max(0, Math.min(Math.floor(index), entries.length));
}
