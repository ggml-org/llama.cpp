/**
 * search_chats core: parameter parsing, matching and formatting. Data comes through a loader, so this has no IndexedDB access.
 * Only user and assistant text on the visible branch is searched and shown, so tool output (e.g. older search results) cannot match.
 */

import { filterByLeafNodeId } from './branching';
import { localIsoString } from './get-datetime';
import {
	SEARCH_CHATS_CUT_NOTICE,
	SEARCH_CHATS_DATE_REGEX,
	SEARCH_CHATS_DATE_TIME_REGEX,
	SEARCH_CHATS_ELLIPSIS,
	SEARCH_CHATS_EXCERPT_LEAD_RATIO,
	SEARCH_CHATS_EXCERPT_RECENTER_RATIO,
	SEARCH_CHATS_FIELD_SEPARATOR,
	SEARCH_CHATS_ID_MAX_CHARS,
	SEARCH_CHATS_LIMIT_CHATS_DEFAULT,
	SEARCH_CHATS_LIMIT_CHATS_MAX,
	SEARCH_CHATS_LIMIT_MESSAGES_DEFAULT,
	SEARCH_CHATS_LIMIT_MESSAGES_MAX,
	SEARCH_CHATS_MAX_IDS,
	SEARCH_CHATS_MAX_PATTERNS,
	SEARCH_CHATS_MESSAGE_CHARS_DEFAULT,
	SEARCH_CHATS_MESSAGE_CHARS_MAX,
	SEARCH_CHATS_MESSAGE_CHARS_MIN,
	SEARCH_CHATS_OUTPUT_MAX_CHARS,
	SEARCH_CHATS_PATTERN_MAX_CHARS
} from '$lib/constants';
import { MessageRole, SearchChatsMessageSelection, SearchChatsOrder } from '$lib/enums';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

export class SearchChatsError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'SearchChatsError';
	}
}

export interface SearchChatsParams {
	byConvIds?: string[];
	byMatches?: string[];
	/** Inclusive, ms since epoch */
	fromTimestamp?: number;
	/** Exclusive, ms since epoch */
	toTimestamp?: number;
	order: SearchChatsOrder;
	limitChats: number;
	offsetChats: number;
	limitMessages: number;
	messageSelection: SearchChatsMessageSelection;
	maxMessageChars: number;
	includeCurrent: boolean;
}

export interface SearchChatsLoader {
	/** Messages of the given conversations, or of all conversations for `null` */
	messages(convIds: string[] | null): Promise<DatabaseMessage[]>;
	/** Ids of conversations with any message in [from, to) */
	conversationIdsInRange(from?: number, to?: number): Promise<Set<string>>;
}

function stringList(
	raw: unknown,
	name: string,
	maxEntries: number,
	maxChars: number
): string[] | undefined {
	if (raw === undefined || raw === null) return undefined;

	const list = typeof raw === 'string' ? [raw] : raw;

	if (!Array.isArray(list) || !list.every((v) => typeof v === 'string' && v.trim() !== '')) {
		throw new SearchChatsError(`${name} must be an array of non-empty strings`);
	}

	if (list.length === 0) return undefined;

	if (list.length > maxEntries) {
		throw new SearchChatsError(`${name} takes at most ${maxEntries} entries`);
	}

	if (list.some((v) => v.length > maxChars)) {
		throw new SearchChatsError(`${name} entries must be at most ${maxChars} characters`);
	}

	return list.map((v) => v.trim());
}

function integer(raw: unknown, name: string, min: number, max: number, fallback: number): number {
	if (raw === undefined || raw === null || raw === '') return fallback;

	const n = Number(raw);

	if (!Number.isInteger(n) || n < min || n > max) {
		throw new SearchChatsError(`${name} must be an integer from ${min} to ${max}`);
	}

	return n;
}

function oneOf<T extends string>(raw: unknown, name: string, options: T[], fallback: T): T {
	if (raw === undefined || raw === null || raw === '') return fallback;

	const value = String(raw).toLowerCase() as T;

	if (!options.includes(value)) {
		throw new SearchChatsError(`${name} must be one of: ${options.join(', ')}`);
	}

	return value;
}

/** ISO 8601 date or date-time; without an offset it is the user's local time */
function timestamp(raw: unknown, name: string): number | undefined {
	if (raw === undefined || raw === null || raw === '') return undefined;

	const error = new SearchChatsError(
		`${name} must be an ISO 8601 date or date-time, e.g. 2026-11-10 or 2026-11-10T14:30 (local time unless an offset is given)`
	);

	if (typeof raw !== 'string') throw error;

	const value = raw.trim().replace(' ', 'T');

	let ms = NaN;

	if (SEARCH_CHATS_DATE_REGEX.test(value)) {
		// Date.parse reads a date-only string as UTC midnight, the user means local midnight
		const [year, month, day] = value.split('-').map(Number);
		const date = new Date(year, month - 1, day);

		if (date.getMonth() === month - 1 && date.getDate() === day) ms = date.getTime();
	} else if (SEARCH_CHATS_DATE_TIME_REGEX.test(value)) {
		ms = Date.parse(value);
	}

	if (Number.isNaN(ms)) throw error;

	return ms;
}

/** Throws SearchChatsError with a message for the model */
export function parseSearchChatsParams(raw: Record<string, unknown>): SearchChatsParams {
	const byConvIds = stringList(
		raw.by_conv_ids,
		'by_conv_ids',
		SEARCH_CHATS_MAX_IDS,
		SEARCH_CHATS_ID_MAX_CHARS
	);
	const byMatches = stringList(
		raw.by_matches,
		'by_matches',
		SEARCH_CHATS_MAX_PATTERNS,
		SEARCH_CHATS_PATTERN_MAX_CHARS
	);
	const fromTimestamp = timestamp(raw.from_timestamp, 'from_timestamp');
	const toTimestamp = timestamp(raw.to_timestamp, 'to_timestamp');

	if (fromTimestamp !== undefined && toTimestamp !== undefined && toTimestamp <= fromTimestamp) {
		throw new SearchChatsError('to_timestamp must be after from_timestamp');
	}

	const defaultSelection = byMatches
		? SearchChatsMessageSelection.MATCHING
		: fromTimestamp !== undefined
			? SearchChatsMessageSelection.EARLIEST
			: SearchChatsMessageSelection.LATEST;
	const messageSelection = oneOf(
		raw.message_selection,
		'message_selection',
		Object.values(SearchChatsMessageSelection),
		defaultSelection
	);

	if (messageSelection === SearchChatsMessageSelection.MATCHING && !byMatches) {
		throw new SearchChatsError(
			`message_selection "${SearchChatsMessageSelection.MATCHING}" needs by_matches; use "${SearchChatsMessageSelection.LATEST}" or "${SearchChatsMessageSelection.EARLIEST}"`
		);
	}

	return {
		byConvIds,
		byMatches,
		fromTimestamp,
		includeCurrent: raw.include_current === true || raw.include_current === 'true',
		limitChats: integer(
			raw.limit_chats,
			'limit_chats',
			1,
			SEARCH_CHATS_LIMIT_CHATS_MAX,
			SEARCH_CHATS_LIMIT_CHATS_DEFAULT
		),
		limitMessages: integer(
			raw.limit_messages,
			'limit_messages',
			0,
			SEARCH_CHATS_LIMIT_MESSAGES_MAX,
			SEARCH_CHATS_LIMIT_MESSAGES_DEFAULT
		),
		maxMessageChars: integer(
			raw.max_message_chars,
			'max_message_chars',
			SEARCH_CHATS_MESSAGE_CHARS_MIN,
			SEARCH_CHATS_MESSAGE_CHARS_MAX,
			SEARCH_CHATS_MESSAGE_CHARS_DEFAULT
		),
		messageSelection,
		offsetChats: integer(raw.offset_chats, 'offset_chats', 0, Number.MAX_SAFE_INTEGER, 0),
		order: oneOf(
			raw.order_timestamp,
			'order_timestamp',
			Object.values(SearchChatsOrder),
			SearchChatsOrder.DESC
		),
		toTimestamp
	};
}

/** SQL LIKE (`%` any run, `_` one char) without regex, so a pattern cannot backtrack exponentially */
function like(text: string, pattern: string): boolean {
	let si = 0;
	let pi = 0;
	let star = -1;
	let mark = 0;

	while (si < text.length) {
		if (
			pi < pattern.length &&
			(pattern[pi] === '_' || (pattern[pi] !== '%' && pattern[pi] === text[si]))
		) {
			si++;
			pi++;
		} else if (pi < pattern.length && pattern[pi] === '%') {
			star = pi++;
			mark = si;
		} else if (star !== -1) {
			pi = star + 1;
			si = ++mark;
		} else {
			return false;
		}
	}

	while (pi < pattern.length && pattern[pi] === '%') pi++;

	return pi === pattern.length;
}

interface Matcher {
	test(text: string): boolean;
	/** Offset in `text` of the longest literal part of the first matching pattern, -1 if none */
	locate(text: string): number;
}

function buildMatcher(patterns: string[]): Matcher {
	// a pattern without % means "contains"
	const compiled = patterns.map((p) => {
		const lower = p.toLowerCase();
		const pattern = lower.includes('%') ? lower : `%${lower}%`;
		const needle = pattern
			.split(/[%_]+/)
			.reduce((best, part) => (part.length > best.length ? part : best), '');

		return { needle, pattern };
	});

	return {
		locate: (text) => {
			const lower = text.toLowerCase();

			// lowercasing can change the length (e.g. U+0130), then offsets do not map back
			if (lower.length !== text.length) return -1;

			const hit = compiled.find(({ pattern }) => like(lower, pattern));

			if (!hit) return -1;

			return hit.needle ? lower.indexOf(hit.needle) : 0;
		},
		test: (text) => {
			const lower = text.toLowerCase();

			return compiled.some(({ pattern }) => like(lower, pattern));
		}
	};
}

/** User and assistant text on the visible branch, in chat order */
function visibleMessages(
	conv: DatabaseConversation,
	messages: DatabaseMessage[]
): DatabaseMessage[] {
	if (messages.length === 0) return [];

	return filterByLeafNodeId(messages, conv.currNode ?? '').filter(
		(m) =>
			(m.role === MessageRole.USER || m.role === MessageRole.ASSISTANT) &&
			!m.isSynthetic &&
			typeof m.content === 'string' &&
			m.content.trim() !== ''
	);
}

function formatDate(ms: number): string {
	const date = new Date(ms);

	return Number.isNaN(date.getTime()) ? 'unknown' : localIsoString(date);
}

function excerpt(text: string, maxChars: number, matchAt: number): string {
	if (text.length <= maxChars) return text;

	if (matchAt > maxChars * SEARCH_CHATS_EXCERPT_RECENTER_RATIO) {
		const lead = Math.floor(maxChars * SEARCH_CHATS_EXCERPT_LEAD_RATIO);
		const start = Math.max(0, Math.min(matchAt - lead, text.length - maxChars));
		const end = start + maxChars;
		const before = start > 0 ? SEARCH_CHATS_ELLIPSIS : '';
		const after = end < text.length ? `${SEARCH_CHATS_ELLIPSIS}[+${text.length - end} chars]` : '';

		return `${before}${text.slice(start, end)}${after}`;
	}

	return `${text.slice(0, maxChars)}${SEARCH_CHATS_ELLIPSIS}[+${text.length - maxChars} chars]`;
}

interface ChatHit {
	conv: DatabaseConversation;
	/** Visible messages of the whole chat, positions are counted on these */
	messages: DatabaseMessage[];
	/** Visible messages inside the time window (all of them without one) */
	inRange: DatabaseMessage[];
	titleMatched: boolean;
	matched: DatabaseMessage[];
}

function formatMessage(
	message: DatabaseMessage,
	position: number,
	params: SearchChatsParams,
	matcher: Matcher | null
): string {
	const text = message.content.trim();
	const matchAt = matcher ? matcher.locate(text) : -1;
	const body = excerpt(text, params.maxMessageChars, matchAt).replace(/\n+/g, '\n  ');
	const files = (message.extra ?? []).map((e) => e.name).filter(Boolean);
	const attachments = files.length > 0 ? ` [attachments: ${files.join(', ')}]` : '';

	return `#${position} ${message.role}${SEARCH_CHATS_FIELD_SEPARATOR}${formatDate(message.timestamp)}: ${body}${attachments}`;
}

function formatChat(
	hit: ChatHit,
	index: number,
	params: SearchChatsParams,
	matcher: Matcher | null,
	timed: boolean
): string {
	const { conv, inRange, matched, messages, titleMatched } = hit;
	const summary = [`conv_id: ${conv.id}`, `updated ${formatDate(conv.lastModified)}`];

	if (params.limitMessages === 0) {
		return `- "${conv.name}"${SEARCH_CHATS_FIELD_SEPARATOR}${summary.join(SEARCH_CHATS_FIELD_SEPARATOR)}`;
	}

	const counts = timed
		? `${messages.length} messages, ${inRange.length} in the time range`
		: `${messages.length} messages`;
	const lines = [
		`## Chat ${index}: "${conv.name}"`,
		[...summary, counts].join(SEARCH_CHATS_FIELD_SEPARATOR)
	];
	const n = params.limitMessages;

	let picked: DatabaseMessage[];

	if (params.messageSelection === SearchChatsMessageSelection.MATCHING && matched.length > 0) {
		const where = [titleMatched ? 'title' : '', `${matched.length} message(s)`]
			.filter(Boolean)
			.join(', ');

		picked = matched.slice(-n);
		lines.push(`matched in: ${where}${matched.length > n ? ` (showing the last ${n})` : ''}`);
	} else {
		if (params.messageSelection === SearchChatsMessageSelection.MATCHING) {
			lines.push('matched in: title only, showing the latest messages');
		}

		picked =
			params.messageSelection === SearchChatsMessageSelection.EARLIEST
				? inRange.slice(0, n)
				: inRange.slice(-n);
	}

	if (messages.length === 0) lines.push('(no messages)');

	const position = new Map(messages.map((m, i) => [m.id, i + 1]));

	for (const m of picked) lines.push(formatMessage(m, position.get(m.id)!, params, matcher));

	return lines.join('\n');
}

function groupByConversation(messages: DatabaseMessage[]): Map<string, DatabaseMessage[]> {
	const groups = new Map<string, DatabaseMessage[]>();

	for (const m of messages) {
		const group = groups.get(m.convId);

		if (group) group.push(m);
		else groups.set(m.convId, [m]);
	}

	return groups;
}

function describeQuery(params: SearchChatsParams): string {
	const parts: string[] = [];

	if (params.byMatches) parts.push(`matching ${JSON.stringify(params.byMatches)}`);

	if (params.byConvIds) parts.push('by conv_id');

	if (params.fromTimestamp !== undefined) parts.push(`from ${formatDate(params.fromTimestamp)}`);

	if (params.toTimestamp !== undefined) parts.push(`before ${formatDate(params.toTimestamp)}`);

	return parts.join(' ');
}

/**
 * Run a search and render the result as text for the model.
 * `conversations` holds the candidates: all conversations, or only those of by_conv_ids.
 */
export async function searchChats(
	conversations: DatabaseConversation[],
	params: SearchChatsParams,
	loader: SearchChatsLoader,
	currentConversationId?: string
): Promise<string> {
	const { byConvIds, fromTimestamp, toTimestamp } = params;
	const timed = fromTimestamp !== undefined || toTimestamp !== undefined;
	const notes: string[] = [];

	let candidates = conversations;

	if (byConvIds) {
		const found = new Set(candidates.map((c) => c.id));
		const missing = byConvIds.filter((id) => !found.has(id));

		if (missing.length > 0) notes.push(`not found: ${missing.join(', ')}`);

		candidates = candidates.filter((c) => byConvIds.includes(c.id));
	}

	if (currentConversationId && !params.includeCurrent) {
		if (byConvIds?.includes(currentConversationId)) {
			notes.push(
				'the current chat is hidden (you already see it); set include_current=true to read it'
			);
		}

		candidates = candidates.filter((c) => c.id !== currentConversationId);
	}

	if (timed) {
		const active = await loader.conversationIdsInRange(fromTimestamp, toTimestamp);

		candidates = candidates.filter((c) => active.has(c.id));
	}

	const matcher = params.byMatches ? buildMatcher(params.byMatches) : null;
	const isInRange = (m: DatabaseMessage) =>
		(fromTimestamp === undefined || m.timestamp >= fromTimestamp) &&
		(toTimestamp === undefined || m.timestamp < toTimestamp);
	const toHit = (conv: DatabaseConversation, raw: DatabaseMessage[]): ChatHit => {
		const messages = visibleMessages(conv, raw);
		const inRange = timed ? messages.filter(isInRange) : messages;
		const matched = matcher ? inRange.filter((m) => matcher.test(m.content)) : [];

		return { conv, inRange, matched, messages, titleMatched: !!matcher?.test(conv.name ?? '') };
	};

	let hits: ChatHit[];

	if (matcher || timed) {
		// which chats qualify depends on their messages, so read them before paging
		const ids = byConvIds || timed ? candidates.map((c) => c.id) : null;
		const byConv = groupByConversation(await loader.messages(ids));

		hits = [];

		for (const conv of candidates) {
			const raw = byConv.get(conv.id) ?? [];

			// cheap check on all messages before building the visible branch
			if (
				matcher &&
				!matcher.test(conv.name ?? '') &&
				!raw.some((m) => typeof m.content === 'string' && matcher.test(m.content))
			) {
				continue;
			}

			const hit = toHit(conv, raw);

			// with a time window, a chat needs visible messages inside it, a title match is not enough
			if (timed && hit.inRange.length === 0) continue;

			if (matcher && !hit.titleMatched && hit.matched.length === 0) continue;

			hits.push(hit);
		}
	} else {
		hits = candidates.map((conv) => toHit(conv, []));
	}

	const sign = params.order === SearchChatsOrder.ASC ? 1 : -1;

	hits.sort((a, b) => sign * (a.conv.lastModified - b.conv.lastModified));

	const total = hits.length;
	const page = hits.slice(params.offsetChats, params.offsetChats + params.limitChats);

	// plain listing or reading: only the chats on this page need their messages
	if (!matcher && !timed && params.limitMessages > 0 && page.length > 0) {
		const byConv = groupByConversation(await loader.messages(page.map((h) => h.conv.id)));

		page.forEach((hit, i) => (page[i] = toHit(hit.conv, byConv.get(hit.conv.id) ?? [])));
	}

	const orderLabel = params.order === SearchChatsOrder.ASC ? 'oldest first' : 'newest first';
	const range =
		page.length === 0
			? `showing none of ${total}`
			: `showing ${params.offsetChats + 1}-${params.offsetChats + page.length} of ${total}`;
	const header = [
		`${`${total} chat(s) ${describeQuery(params)}`.trim()} (${orderLabel}, ${range})`,
		...notes
	].join('\n');
	const separator = params.limitMessages === 0 ? '\n' : '\n\n';
	const moreNote = (shown: number) =>
		`[${page.length - shown} more chat(s) on this page not shown: use offset_chats=${params.offsetChats + shown}, or lower limit_messages]`;
	// room for the note in its longest form
	const budget = SEARCH_CHATS_OUTPUT_MAX_CHARS - separator.length - moreNote(0).length;
	const blocks: string[] = [];

	let size = header.length;

	for (const [i, hit] of page.entries()) {
		const block = formatChat(hit, params.offsetChats + i + 1, params, matcher, timed);
		const isLast = i === page.length - 1;
		const limit = isLast ? SEARCH_CHATS_OUTPUT_MAX_CHARS : budget;

		if (size + separator.length + block.length > limit) {
			// always show at least part of the first chat
			if (blocks.length === 0) {
				const room = limit - size - separator.length - SEARCH_CHATS_CUT_NOTICE.length;

				blocks.push(`${block.slice(0, Math.max(0, room))}${SEARCH_CHATS_CUT_NOTICE}`);
			}

			break;
		}

		blocks.push(block);
		size += separator.length + block.length;
	}

	if (blocks.length < page.length) blocks.push(moreNote(blocks.length));

	return blocks.length > 0 ? `${header}\n\n${blocks.join(separator)}` : header;
}
