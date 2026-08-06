/**
 * Tokenizer for the chat-form contenteditable input: maps between the
 * markdown source and the badge/text token stream the DOM is built from.
 *
 * A badge is one opaque source contribution (`[name](file://path)`); only
 * `root.childNodes` is ever walked, never a badge's own subtree (its label
 * length is not its source length). The caret cannot land inside a badge
 * (`contenteditable=false`), so offsets resolve to the nearest badge edge.
 */

import {
	decodeFileLinkPath,
	fileMentionLinkRe,
	getMentionBadgeIconPaths,
	getMentionBadgeLabel
} from './mention-badge';
import {
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	MENTION_BADGE_SVG_ATTRIBUTES,
	SETTINGS_KEYS
} from '$lib/constants';
import { settingsStore } from '$lib/stores/settings.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';

export type ContentToken =
	| { kind: 'text'; text: string }
	| { kind: 'badge'; name: string; path: string };

// Recognize completed `[name](file://path)` insertions. `file://` is
// required so plain URLs stay as text; `)` is allowed only when not
// followed by whitespace or `[` (adjacent badges keep terminating).
const MENTION_BADGE_RE = fileMentionLinkRe('g');

/**
 * Compute the byte-length contribution of one badge in source form.
 * Centralized so `serializeContent`, `rangeToTextOffset` and
 * `textOffsetToRange` agree on what counts; otherwise math of
 * `caret offset -> markdown offset` silently breaks.
 */
function badgeSourceLength(name: string, path: string): number {
	if (!name || !path) return 0;
	return `[${name}](file://${path})`.length;
}

/**
 * Tokenize a markdown source value into the segments the
 * contenteditable will render. Plain text and badges interleave in
 * source order. Any whitespace after a badge stays in a plain
 * text token so the round trip is byte-exact.
 */
export function tokenizeContent(input: string): ContentToken[] {
	const tokens: ContentToken[] = [];
	let cursor = 0;
	MENTION_BADGE_RE.lastIndex = 0;

	let match: RegExpExecArray | null;
	while ((match = MENTION_BADGE_RE.exec(input)) !== null) {
		const [whole, name, path] = match;
		const start = match.index;

		if (start > cursor) {
			tokens.push({ kind: 'text', text: input.slice(cursor, start) });
		}

		tokens.push({ kind: 'badge', name, path });
		cursor = start + whole.length;
	}

	if (cursor < input.length) {
		tokens.push({ kind: 'text', text: input.slice(cursor) });
	}

	return tokens;
}

/**
 * Serialize a contenteditable subtree back to source. Only direct
 * `childNodes` are walked (a badge is one opaque contribution);
 * non-text, non-badge nodes are skipped so browser-injected
 * wrappers do not leak into the source.
 */
export function serializeContent(root: HTMLElement): string {
	let out = '';

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			out += child.textContent ?? '';
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;
		if (el.dataset.mentionBadge !== 'true') continue;

		const name = el.dataset.mentionName ?? '';
		const path = el.dataset.mentionPath ?? '';
		if (name && path) {
			out += `[${name}](file://${path})`;
		}
	}

	return out;
}

/**
 * Compute the plain-text character offset of a `Range` anchored
 * inside the contenteditable root. Used to capture caret position
 * before any DOM rebuild so we can restore it after.
 *
 * If `range` is null (selection lost during teardown) the position
 * falls back to buffer length. The body walks `tmp.childNodes`
 * only, so badges contribute their full source length, not their
 * visible label width. `cloneContents()` truncates the trailing
 * text node properly via the browser's range semantics, so its
 * `textContent` is the buffer length up to and including the caret.
 */
export function rangeToTextOffset(root: HTMLElement, range: Range | null): number {
	if (!range) return serializeContent(root).length;

	const pre = range.cloneRange();
	pre.selectNodeContents(root);
	pre.setEnd(range.endContainer, range.endOffset);

	const tmp = document.createElement('div');
	tmp.appendChild(pre.cloneContents());

	let total = 0;
	for (const child of Array.from(tmp.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			total += (child.textContent ?? '').length;
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;
		if (el.dataset.mentionBadge !== 'true') continue;

		total += badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
	}

	return total;
}

/**
 * Materialize a single token stream into a freshly-built DOM subtree
 * suitable for inserting in place of the live contenteditable body.
 * The returned fragment contains plain text nodes for text tokens
 * and `<span data-mention-badge="true">` elements for badges. The
 * badge's class string + inline folder SVG mirror
 * `MentionBadge.svelte` exactly; Tailwind scans both and gets the
 * same style applied.
 */
export function buildFragment(tokens: ContentToken[]): DocumentFragment {
	const fragment = document.createDocumentFragment();

	for (const token of tokens) {
		if (token.kind === 'text') {
			fragment.appendChild(document.createTextNode(token.text));
			continue;
		}

		// A leading badge gets an empty text node prepended: without a
		// real text position at the buffer start, the spot before the
		// badge is unreachable via keyboard (ArrowLeft/Home). The empty
		// node serializes to nothing, so the round trip stays byte-exact.
		if (!fragment.lastChild) {
			fragment.appendChild(document.createTextNode(''));
		}

		const badge = document.createElement('span');
		badge.dataset.mentionBadge = 'true';
		badge.dataset.mentionName = token.name;
		badge.dataset.mentionPath = token.path;
		badge.title = decodeFileLinkPath(token.path);
		badge.className = MENTION_BADGE_CLASSNAME;
		badge.contentEditable = 'false';

		// Icon - matches the lucide component picked by MentionBadge.svelte
		// so the DOM-built badge is visually identical.
		const svg = document.createElementNS(MENTION_BADGE_SVG_ATTRIBUTES['xmlns'], 'svg');
		for (const [attr, value] of Object.entries(MENTION_BADGE_SVG_ATTRIBUTES)) {
			svg.setAttribute(attr, value);
		}
		for (const cls of MENTION_BADGE_ICON_CLASSNAME.split(/\s+/).filter(Boolean)) {
			svg.classList.add(cls);
		}

		for (const d of getMentionBadgeIconPaths(token.path)) {
			const path = document.createElementNS(MENTION_BADGE_SVG_ATTRIBUTES['xmlns'], 'path');
			path.setAttribute('d', d);
			svg.appendChild(path);
		}

		const label = document.createElement('span');
		label.classList.add('shrink-0', 'truncate');
		label.textContent = getMentionBadgeLabel(
			token.name,
			decodeFileLinkPath(token.path),
			settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS),
			toolsStore.serverHome
		);

		badge.appendChild(svg);
		badge.appendChild(label);
		fragment.appendChild(badge);
	}

	return fragment;
}

const WORD_CHAR_RE = /[\p{L}\p{N}_]/u;

/**
 * Word-jump target (Option+Arrow / Ctrl+Arrow) in source offsets, or
 * null when the jump crosses no badge and native word movement should
 * handle it. Badge spans are masked to word characters and act as
 * hard word-run boundaries, so a badge counts as exactly one word:
 * native word iteration treats the non-editable badge element
 * inconsistently and overshoots it by a full word in either direction.
 */
export function badgeAwareWordJump(
	source: string,
	offset: number,
	direction: 'forward' | 'backward'
): number | null {
	let masked = '';
	const badgeSpans: Array<[number, number]> = [];

	for (const token of tokenizeContent(source)) {
		const len =
			token.kind === 'badge' ? badgeSourceLength(token.name, token.path) : token.text.length;
		if (token.kind === 'badge') badgeSpans.push([masked.length, masked.length + len]);
		masked += token.kind === 'badge' ? 'a'.repeat(len) : token.text;
	}

	if (badgeSpans.length === 0) return null;

	const isWord = (index: number) => WORD_CHAR_RE.test(masked[index]);
	const spanStartingAt = (index: number) => badgeSpans.find(([start]) => start === index);
	const spanEndingAt = (index: number) => badgeSpans.find(([, end]) => end === index);
	const n = masked.length;
	let i = offset;

	if (direction === 'forward') {
		// Skip non-word run when starting outside a word, then skip the
		// word run itself. Entering a badge completes the word phase at
		// the badge's end edge.
		if (!(i < n && isWord(i))) {
			while (i < n && !isWord(i)) i++;
		}
		while (i < n && isWord(i)) {
			const span = spanStartingAt(i);
			if (span) {
				i = span[1];
				break;
			}
			i++;
		}
	} else {
		if (!(i > 0 && isWord(i - 1))) {
			while (i > 0 && !isWord(i - 1)) i--;
		}
		while (i > 0 && isWord(i - 1)) {
			const span = spanEndingAt(i);
			if (span) {
				i = span[0];
				break;
			}
			i--;
		}
	}

	if (i === offset) return null;

	const lo = Math.min(offset, i);
	const hi = Math.max(offset, i);
	return badgeSpans.some(([start, end]) => start < hi && end > lo) ? i : null;
}

/**
 * Returns 0 when `caret` sits exactly at a leading badge's end edge,
 * null otherwise. Plain ArrowLeft at that spot has no native previous
 * position (the buffer starts with a non-editable element), so the
 * host snaps the caret to the buffer start manually. Covers post-edit
 * states where the leading pad from `buildFragment` is gone (e.g. the
 * user deleted the text before a mid-text badge).
 */
export function leadingBadgeEdgeOffset(source: string, caret: number): number | null {
	const [first] = tokenizeContent(source);
	if (!first || first.kind !== 'badge') return null;
	return caret === badgeSourceLength(first.name, first.path) ? 0 : null;
}

/**
 * Translate a plain-text character offset into a `Range` placed at
 * that position in the DOM. Returns a degenerate range (collapsed
 * to a single point). Out-of-range `offset` clamps to buffer end.
 *
 * Inside a badge we cannot land caret, so the offset resolves to
 * one of the two badge edges: zero offset lands BEFORE the badge,
 * any positive source offset lands AFTER. This matches the
 * visible-edit behavior the user expects from a non-editable
 * inline element.
 */
export function textOffsetToRange(root: HTMLElement, offset: number): Range {
	const range = document.createRange();
	let remaining = offset;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			const text = child.textContent ?? '';
			if (remaining <= text.length) {
				range.setStart(child, remaining);
				range.setEnd(child, remaining);
				return range;
			}
			remaining -= text.length;
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;
		if (el.dataset.mentionBadge !== 'true') continue;

		const badgeLen = badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
		if (remaining <= badgeLen) {
			if (remaining === 0) {
				range.setStartBefore(el);
				range.setEndBefore(el);
			} else {
				range.setStartAfter(el);
				range.setEndAfter(el);
			}
			return range;
		}
		remaining -= badgeLen;
	}

	range.selectNodeContents(root);
	range.collapse(false);
	return range;
}
