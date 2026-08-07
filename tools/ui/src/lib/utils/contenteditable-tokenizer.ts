/**
 * Maps between the chat-form contenteditable's markdown source and the
 * badge/text token stream the DOM is built from. A badge is one opaque
 * source contribution (`[name](file://path)`); a badge's own subtree is
 * never walked (its label length is not its source length). The caret
 * cannot land inside a badge, so offsets resolve to the nearest badge
 * edge.
 *
 * The tokenizer emits a flat DOM (text nodes + badges), but browsers
 * restructure it on Enter: Chromium appends `<div>` line wrappers,
 * Firefox wraps the whole buffer in them, and Shift+Enter / mobile
 * keyboards / execCommand produce `<br>` shapes. Serialization folds
 * those back into `\n` so the source never diverges from what is on
 * screen, and both offset mappers understand the same shapes.
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

// Block wrappers browsers insert for newlines (Chromium/Firefox use
// `<div>`); each one folds back into a single `\n` during serialization.
const BLOCK_TAG_NAMES = new Set(['DIV', 'P']);

// Recognize completed `[name](file://path)` insertions. `file://` is
// required so plain URLs stay as text; `)` is allowed only when not
// followed by whitespace or `[` (adjacent badges keep terminating).
const MENTION_BADGE_RE = fileMentionLinkRe('g');

/**
 * Source-form length of one badge, shared by the offset math so
 * `serializeContent`, `rangeToTextOffset` and `textOffsetToRange` agree.
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
 * Serialize a contenteditable subtree back to source. A badge is one
 * opaque contribution; `<br>` and block wrappers the browser inserted
 * for newlines fold back into `\n` (a trailing `<br>` is the browser's
 * caret placeholder, not a newline). Any other element is transparent.
 */
export function serializeContent(root: HTMLElement): string {
	let out = '';

	const walk = (parent: Node) => {
		let first = true; // no source-contributing sibling seen yet

		for (const child of Array.from(parent.childNodes)) {
			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';
				if (text.length > 0) {
					out += text;
					first = false;
				}
				continue;
			}

			if (child.nodeType !== Node.ELEMENT_NODE) continue;

			const el = child as HTMLElement;

			if (el.dataset.mentionBadge === 'true') {
				const name = el.dataset.mentionName ?? '';
				const path = el.dataset.mentionPath ?? '';
				if (name && path) {
					out += `[${name}](file://${path})`;
					first = false;
				}
				continue;
			}

			if (el.tagName === 'BR') {
				if (el.nextSibling) {
					out += '\n';
					first = false;
				}
				continue;
			}

			if (BLOCK_TAG_NAMES.has(el.tagName)) {
				if (!first) out += '\n';
				walk(el);
				first = false;
				continue;
			}

			const before = out.length;
			walk(el);
			if (out.length > before) first = false;
		}
	};

	walk(root);
	return out;
}

/**
 * Plain-text offset of a `Range` in the root, so the caret can be restored
 * after a DOM rebuild. Null range (selection lost) falls back to buffer
 * length. Badges count their full source length, not their label width.
 * Walked against the live DOM (not a clone) so a `<br>` keeps its
 * trailing/not-trailing context.
 */
export function rangeToTextOffset(root: HTMLElement, range: Range | null): number {
	if (!range) return serializeContent(root).length;

	// A point is at/before the caret iff it falls inside [root start, caret].
	const pre = range.cloneRange();
	pre.selectNodeContents(root);
	pre.setEnd(range.endContainer, range.endOffset);
	const atOrBeforeCaret = (node: Node, offset: number) => pre.comparePoint(node, offset) !== 1;

	let total = 0;
	let done = false;

	const walk = (parent: Node) => {
		let first = true;

		for (const child of Array.from(parent.childNodes)) {
			if (done) return;

			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';
				if (text.length === 0) continue;
				if (!atOrBeforeCaret(child, 0)) {
					done = true;
					return;
				}
				if (range.endContainer === child) {
					total += range.endOffset;
					done = true;
					return;
				}
				total += text.length;
				first = false;
				continue;
			}

			if (child.nodeType !== Node.ELEMENT_NODE) continue;

			const el = child as HTMLElement;
			const parentNode = el.parentNode as Node;
			const elIndex = Array.prototype.indexOf.call(parentNode.childNodes, el);

			if (el.dataset.mentionBadge === 'true') {
				const len = badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
				if (len === 0) continue;
				if (!atOrBeforeCaret(parentNode, elIndex + 1)) {
					done = true;
					return;
				}
				total += len;
				first = false;
				continue;
			}

			if (el.tagName === 'BR') {
				if (!el.nextSibling) continue;
				if (!atOrBeforeCaret(parentNode, elIndex + 1)) {
					done = true;
					return;
				}
				total += 1;
				first = false;
				continue;
			}

			if (BLOCK_TAG_NAMES.has(el.tagName)) {
				if (!first) {
					if (!atOrBeforeCaret(el, 0)) {
						done = true;
						return;
					}
					total += 1;
				}
				walk(el);
				first = false;
				continue;
			}

			const before = total;
			walk(el);
			if (total > before) first = false;
		}
	};

	walk(root);
	return total;
}

/**
 * Materialize a token stream into a DOM subtree for the contenteditable
 * body: text nodes for text tokens, `<span data-mention-badge="true">`
 * elements for badges. The badge's class string + inline SVG are shared
 * with the rehype plugin via `$lib/constants/mention-badge`.
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
 * Word-jump target (Option+Arrow / Ctrl+Arrow) in source offsets, or null
 * when the jump crosses no badge and native word movement should handle it.
 * Badge spans are masked to word characters and act as hard word-run
 * boundaries, so a badge counts as exactly one word.
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
 * Returns 0 when `caret` sits exactly at a leading badge's end edge, null
 * otherwise. Plain ArrowLeft there has no native previous position (the
 * buffer starts with a non-editable element), so the host snaps the caret
 * to the buffer start manually.
 */
export function leadingBadgeEdgeOffset(source: string, caret: number): number | null {
	const [first] = tokenizeContent(source);
	if (!first || first.kind !== 'badge') return null;
	return caret === badgeSourceLength(first.name, first.path) ? 0 : null;
}

/**
 * Translate a plain-text offset into a degenerate `Range` at that position
 * in the DOM; out-of-range offsets clamp to buffer end. The caret cannot
 * land inside a badge, so zero offset lands BEFORE the badge and any
 * positive source offset lands AFTER it. Understands the same browser
 * block/`<br>` newline shapes as `serializeContent`.
 */
export function textOffsetToRange(root: HTMLElement, offset: number): Range {
	const range = document.createRange();
	let remaining = offset;
	let landed = false;

	const walk = (parent: Node) => {
		let first = true;

		for (const child of Array.from(parent.childNodes)) {
			if (landed) return;

			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';
				if (text.length === 0) continue;
				if (remaining <= text.length) {
					range.setStart(child, remaining);
					range.setEnd(child, remaining);
					landed = true;
					return;
				}
				remaining -= text.length;
				first = false;
				continue;
			}

			if (child.nodeType !== Node.ELEMENT_NODE) continue;

			const el = child as HTMLElement;

			if (el.dataset.mentionBadge === 'true') {
				const len = badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
				if (len === 0) continue;
				if (remaining <= len) {
					if (remaining === 0) {
						range.setStartBefore(el);
						range.setEndBefore(el);
					} else {
						range.setStartAfter(el);
						range.setEndAfter(el);
					}
					landed = true;
					return;
				}
				remaining -= len;
				first = false;
				continue;
			}

			if (el.tagName === 'BR') {
				if (!el.nextSibling) continue;
				if (remaining === 0) {
					range.setStartBefore(el);
					range.setEndBefore(el);
					landed = true;
					return;
				}
				remaining -= 1;
				first = false;
				continue;
			}

			if (BLOCK_TAG_NAMES.has(el.tagName)) {
				if (!first) {
					if (remaining === 0) {
						// The boundary newline belongs to the previous line.
						range.setStartBefore(el);
						range.setEndBefore(el);
						landed = true;
						return;
					}
					remaining -= 1;
				}
				walk(el);
				first = false;
				continue;
			}

			const before = remaining;
			walk(el);
			if (remaining < before) first = false;
		}
	};

	walk(root);

	if (!landed) {
		range.selectNodeContents(root);
		range.collapse(false);
	}

	return range;
}
