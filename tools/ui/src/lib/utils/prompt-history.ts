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

export type PromptHistoryScope = 'separate' | 'combine';

export interface PromptHistoryBuckets {
	combined: string[];
	sessions: Record<string, string[]>;
}

function parseEntryList(value: unknown): string[] {
	if (!Array.isArray(value)) {
		return [];
	}

	return value.filter((item): item is string => typeof item === 'string' && item.trim() !== '');
}

/** Combined + per-session lists. A legacy JSON array becomes `combined`. */
export function parsePromptHistoryBuckets(raw: string | null): PromptHistoryBuckets {
	if (!raw) {
		return { combined: [], sessions: {} };
	}

	try {
		const parsed: unknown = JSON.parse(raw);

		if (Array.isArray(parsed)) {
			return { combined: parseEntryList(parsed), sessions: {} };
		}

		if (!parsed || typeof parsed !== 'object') {
			return { combined: [], sessions: {} };
		}

		const record = parsed as { combined?: unknown; sessions?: unknown };
		const sessions: Record<string, string[]> = {};

		if (record.sessions && typeof record.sessions === 'object' && !Array.isArray(record.sessions)) {
			for (const [id, list] of Object.entries(record.sessions as Record<string, unknown>)) {
				sessions[id] = parseEntryList(list);
			}
		}

		return { combined: parseEntryList(record.combined), sessions };
	} catch {
		return { combined: [], sessions: {} };
	}
}

export function getPromptHistoryEntries(
	store: PromptHistoryBuckets,
	scope: PromptHistoryScope,
	sessionId: string
): string[] {
	if (scope === 'combine') {
		return store.combined;
	}

	return store.sessions[sessionId] ?? [];
}

export function setPromptHistoryEntries(
	store: PromptHistoryBuckets,
	scope: PromptHistoryScope,
	sessionId: string,
	entries: string[]
): PromptHistoryBuckets {
	if (scope === 'combine') {
		return { combined: entries, sessions: store.sessions };
	}

	return {
		combined: store.combined,
		sessions: { ...store.sessions, [sessionId]: entries }
	};
}

export function emptyPromptHistoryBuckets(): PromptHistoryBuckets {
	return { combined: [], sessions: {} };
}

/** User message texts in time order; skips blank and synthetic rows. */
export function userPromptTextsFromMessages(
	messages: { content?: string; isSynthetic?: boolean; role: string; timestamp?: number }[]
): string[] {
	return [...messages]
		.sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
		.filter(
			(message) => message.role === 'user' && !message.isSynthetic && Boolean(message.content?.trim())
		)
		.map((message) => message.content!.trim());
}

/**
 * Append prompts to both the combined list and that session's list so
 * either scope still works after a later settings change.
 */
export function appendImportedPrompts(
	store: PromptHistoryBuckets,
	sessionId: string,
	texts: string[]
): PromptHistoryBuckets {
	let combined = store.combined;
	let session = store.sessions[sessionId] ?? [];

	for (const text of texts) {
		combined = pushPromptHistory(combined, text);
		session = pushPromptHistory(session, text);
	}

	return {
		combined,
		sessions: { ...store.sessions, [sessionId]: session }
	};
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

/** Wheel up (negative deltaY) recalls previous; wheel down recalls next. */
export function wheelHistoryDirection(
	deltaY: number,
	deltaX = 0
): PromptHistorySwipe | null {
	if (!Number.isFinite(deltaY) || Math.abs(deltaY) < 1) {
		return null;
	}

	if (Math.abs(deltaY) <= Math.abs(deltaX)) {
		return null;
	}

	return deltaY < 0 ? 'up' : 'down';
}

/**
 * Lock document scrolling while a finger moves vertically on the chat
 * input, unless the input itself still has room to scroll that way.
 */
export function shouldLockPageScroll(
	dx: number,
	dy: number,
	inputEl?: HTMLElement | null
): boolean {
	if (Math.abs(dy) <= 2 || Math.abs(dy) <= Math.abs(dx)) {
		return false;
	}

	const direction: PromptHistorySwipe = dy < 0 ? 'up' : 'down';

	return !(inputEl && canScrollInDirection(inputEl, direction));
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
