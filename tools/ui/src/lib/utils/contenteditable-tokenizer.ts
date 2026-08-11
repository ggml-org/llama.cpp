/**
 * Maps between the chat-form contenteditable's markdown source and the
 * badge/code/text token stream the DOM is built from. A badge is one
 * opaque source contribution (`[name](file://path)`); its own subtree
 * is never walked, and the caret cannot land inside it, so offsets
 * resolve to the nearest badge edge. Code spans (`<code data-code-token>`)
 * are EDITABLE, unlike badges: they carry the full source segment
 * (backtick fences included) as their text, so their textContent
 * serializes verbatim and source offsets map 1:1 to text offsets.
 *
 * The tokenizer emits a flat DOM (text nodes + badges + code spans),
 * but browsers restructure it on Enter (`<div>` line wrappers, `<br>`
 * shapes). Serialization folds those back into `\n` so the source
 * never diverges from what is on screen; both offset mappers
 * understand the same shapes.
 *
 * The newline separating a fenced block from adjacent content is a
 * SOURCE-level concept, never stored in the DOM: the block is
 * display:block, so a leading `\n` in the following text node would
 * render as a phantom empty line. Serialization synthesizes exactly
 * one `\n` at every block boundary and `buildFragment` strips it from
 * text tokens. A text node's own leading/trailing `\n` next to a
 * block is an ADDITIONAL blank line.
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
	| { kind: 'badge'; name: string; path: string }
	| { kind: 'inlineCode'; text: string }
	| { kind: 'codeBlock'; text: string };

// Block wrappers browsers insert for newlines; each folds back into a
// single `\n` during serialization.
const BLOCK_TAG_NAMES = new Set(['DIV', 'P']);
// `file://` is required so plain URLs stay as text; `)` terminates only
// when not followed by whitespace or `[` (adjacent badges keep working).
const MENTION_BADGE_RE = fileMentionLinkRe('g');

function badgeSourceLength(name: string, path: string): number {
	if (!name || !path) return 0;

	return `[${name}](file://${path})`.length;
}

/**
 * Recognize complete code spans. Fenced blocks (triple backticks,
 * optional language, possibly multiline) take priority over inline
 * spans (single backticks, single line, non-empty). Only CLOSED
 * spans match: an unclosed fence stays plain text until the closing
 * backticks land. The match includes the fences so the token's
 * source length equals its rendered text length.
 */
const CODE_SPAN_RE = /(```[\s\S]*?```)|(`[^`\n]+`)/g;

/**
 * Cheap gate check for `ChatForm`: does the buffer contain a
 * complete code span (inline or fenced)? Used to promote the plain
 * textarea to the contenteditable renderer.
 */
export function containsCodeSpan(value: string): boolean {
	CODE_SPAN_RE.lastIndex = 0;

	return CODE_SPAN_RE.test(value);
}

const CODE_FENCE_RE = /```/g;

/**
 * Is `offset` inside a fenced code block region? Toggle-based: an
 * odd number of ``` fences before the offset means the position
 * sits in block content. Unlike `containsCodeSpan` this also
 * counts the still-OPEN fence while the user is typing a block
 * (no closing ``` yet), so Enter can add a line instead of
 * submitting the message.
 */
export function isOffsetInCodeBlock(source: string, offset: number): boolean {
	let inside = false;

	CODE_FENCE_RE.lastIndex = 0;

	let match: RegExpExecArray | null;

	while ((match = CODE_FENCE_RE.exec(source)) !== null) {
		if (match.index + match[0].length > offset) break;

		inside = !inside;
	}

	return inside;
}

/**
 * Tokenize a markdown source value into the segments the
 * contenteditable will render. Code spans are carved out first
 * (their content is literal - a `file://` link inside backticks
 * must NOT render as a badge), then plain text and badges
 * interleave in the remaining gaps. Any whitespace after a badge
 * stays in a plain text token so the round trip is byte-exact.
 */
export function tokenizeContent(input: string): ContentToken[] {
	const tokens: ContentToken[] = [];

	let cursor = 0;

	CODE_SPAN_RE.lastIndex = 0;

	let match: RegExpExecArray | null;

	while ((match = CODE_SPAN_RE.exec(input)) !== null) {
		const start = match.index;

		if (start > cursor) {
			pushTextAndBadgeTokens(input.slice(cursor, start), tokens);
		}

		tokens.push(
			match[1] !== undefined
				? { kind: 'codeBlock', text: match[1] }
				: { kind: 'inlineCode', text: match[2] }
		);
		cursor = start + match[0].length;
	}

	if (cursor < input.length) {
		pushTextAndBadgeTokens(input.slice(cursor), tokens);
	}

	return tokens;
}

/**
 * Tokenize a code-free segment into text and badge tokens.
 */
function pushTextAndBadgeTokens(input: string, tokens: ContentToken[]) {
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
}

function isCodeBlockElement(node: Node | null): node is HTMLElement {
	return node instanceof HTMLElement && node.dataset.codeToken === 'block';
}

/**
 * Serialize a contenteditable subtree back to source. `<br>` and block
 * wrappers the browser inserted for newlines fold back into `\n` (a
 * trailing `<br>` is the browser's caret placeholder, not a newline);
 * any other element is transparent. Code spans serialize their
 * textContent verbatim (fences included). One separator `\n` is
 * synthesized at every fenced-block boundary (the DOM never stores
 * it), and a `<br>` adjacent to a code block is an escape hatch, not
 * a newline.
 */
export function serializeContent(root: HTMLElement): string {
	let out = '';
	let pendingBlockBoundary = false;

	const walk = (parent: Node) => {
		let first = true; // no source-contributing sibling seen yet

		for (const child of Array.from(parent.childNodes)) {
			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';

				if (text.length > 0) {
					if (pendingBlockBoundary) {
						out += '\n';
						pendingBlockBoundary = false;
					}

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
					if (pendingBlockBoundary) {
						out += '\n';
						pendingBlockBoundary = false;
					}

					out += `[${name}](file://${path})`;
					first = false;
				}

				continue;
			}

			if (el.dataset.codeToken !== undefined) {
				const isBlock = el.dataset.codeToken === 'block';

				if (isBlock && (pendingBlockBoundary || !first)) out += '\n';

				pendingBlockBoundary = false;
				walk(el);
				first = false;

				if (isBlock) pendingBlockBoundary = true;

				continue;
			}

			if (el.tagName === 'BR') {
				const isHatch =
					isCodeBlockElement(el.previousSibling) || isCodeBlockElement(el.nextSibling);

				if (!isHatch && el.nextSibling) {
					if (pendingBlockBoundary) {
						out += '\n';
						pendingBlockBoundary = false;
					}

					out += '\n';
					first = false;
				}

				continue;
			}

			if (BLOCK_TAG_NAMES.has(el.tagName)) {
				if (pendingBlockBoundary || !first) out += '\n';

				pendingBlockBoundary = false;
				walk(el);
				first = false;

				continue;
			}

			walk(el);

			if (pendingBlockBoundary) first = false;
		}
	};

	walk(root);

	return out;
}

/**
 * Compare the live DOM's non-text structure against a token stream.
 * Only element contributions are compared (badges by name/path, code
 * spans by kind and source segment): text nodes are owned by the
 * browser between rebuilds, so their split/merge state is irrelevant.
 * A mismatch means token boundaries shifted (a code span was just
 * completed or broken) and the DOM needs a rebuild to restyle.
 */
export function domMatchesTokens(root: HTMLElement, tokens: ContentToken[]): boolean {
	const expected = tokens.filter((token) => token.kind !== 'text');

	let index = 0;

	const walk = (parent: Node): boolean => {
		for (const child of Array.from(parent.childNodes)) {
			if (child.nodeType !== Node.ELEMENT_NODE) continue;

			const el = child as HTMLElement;
			const isBadge = el.dataset.mentionBadge === 'true';
			const isCode = el.dataset.codeToken !== undefined;

			if (!isBadge && !isCode) {
				if (!walk(el)) return false;

				continue;
			}

			const token = expected[index++];

			if (!token) return false;

			if (isBadge) {
				if (token.kind !== 'badge') return false;

				if (token.name !== (el.dataset.mentionName ?? '')) return false;

				if (token.path !== (el.dataset.mentionPath ?? '')) return false;

				continue;
			}

			const codeKind = el.dataset.codeToken === 'block' ? 'codeBlock' : 'inlineCode';

			if (token.kind !== codeKind) return false;

			if (
				(token.kind === 'inlineCode' || token.kind === 'codeBlock') &&
				token.text !== (el.textContent ?? '')
			) {
				return false;
			}
		}

		return true;
	};

	return walk(root) && index === expected.length;
}

/**
 * Plain-text offset of a `Range` in the root; null range (selection
 * lost) falls back to buffer length. Walked against the live DOM (not
 * a clone) so a `<br>` keeps its trailing/not-trailing context. Code
 * spans count their full textContent (fences included) and the caret
 * may land inside them; synthesized block boundaries count one `\n`
 * once the caret is past them.
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
	// DOM position of a code block's synthesized after-boundary, set
	// when walking past a block and consumed by the next contributing
	// sibling (counts one `\n` once the caret is past it).
	let pendingPoint: { node: Node; index: number } | null = null;

	const walk = (parent: Node) => {
		let first = true;

		for (const child of Array.from(parent.childNodes)) {
			if (done) return;

			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';

				if (text.length === 0) continue;

				if (pendingPoint) {
					const { index, node } = pendingPoint;

					pendingPoint = null;

					if (!atOrBeforeCaret(node, index)) {
						done = true;

						return;
					}

					total += 1;
				}

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

			if (pendingPoint) {
				const { index, node } = pendingPoint;

				pendingPoint = null;

				if (!atOrBeforeCaret(node, index)) {
					done = true;

					return;
				}

				total += 1;
			}

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

			if (el.dataset.codeToken !== undefined) {
				const isBlock = el.dataset.codeToken === 'block';

				if (isBlock && !first) {
					if (!atOrBeforeCaret(el, 0)) {
						done = true;

						return;
					}

					total += 1;
				}

				walk(el);
				first = false;

				if (isBlock) pendingPoint = { index: elIndex + 1, node: parentNode };

				continue;
			}

			if (el.tagName === 'BR') {
				const isHatch =
					isCodeBlockElement(el.previousSibling) || isCodeBlockElement(el.nextSibling);

				if (isHatch || !el.nextSibling) continue;

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
 * Materialize a token stream into a DOM subtree: text nodes for text
 * tokens, `<span data-mention-badge="true">` elements for badges,
 * `<code data-code-token>` elements for code spans. The badge's class
 * string + inline SVG are shared with the rehype plugin via
 * `$lib/constants/mention-badge`.
 */
export function buildFragment(tokens: ContentToken[]): DocumentFragment {
	const fragment = document.createDocumentFragment();

	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];

		if (token.kind === 'text') {
			let text = token.text;

			// The separator \n at a fenced-block boundary is synthesized
			// at serialization time; keeping it in the DOM would render a
			// phantom empty line next to the block.
			if (tokens[index - 1]?.kind === 'codeBlock' && text.startsWith('\n')) {
				text = text.slice(1);
			}

			if (tokens[index + 1]?.kind === 'codeBlock' && text.endsWith('\n')) {
				text = text.slice(0, -1);
			}

			if (text.length === 0) continue;

			fragment.appendChild(document.createTextNode(text));

			continue;
		}

		if (token.kind === 'inlineCode' || token.kind === 'codeBlock') {
			const code = document.createElement('code');

			code.dataset.codeToken = token.kind === 'codeBlock' ? 'block' : 'inline';
			code.textContent = token.text;
			fragment.appendChild(code);

			continue;
		}

		// A leading badge gets an empty text node prepended: without a real
		// text position at the buffer start, the spot before the badge is
		// unreachable via keyboard (ArrowLeft/Home).
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

// A sibling provides a reachable caret line when it is an element
// (badge, another block, an existing hatch) or a non-empty text node.
function hasLineBeside(node: Node | null): boolean {
	if (!node) return false;

	if (node.nodeType === Node.ELEMENT_NODE) return true;

	return (node.textContent ?? '') !== '';
}

/**
 * A code block at the END of the buffer needs an editable line after
 * it: without one the caret cannot leave the block with
 * ArrowDown/ArrowRight. A trailing `<br>` provides that line while
 * staying transparent to serialization (skipped as a hatch), and is
 * removed again once real content takes its place.
 *
 * No hatch is added BEFORE a leading block: the empty line above it
 * is transient and managed by the component (created when the caret
 * arrows onto it, removed when the caret leaves). A transient
 * leading hatch found here is kept; the browser's lone placeholder
 * `<br>` in an empty root is left untouched.
 */
export function syncCodeBlockHatches(root: HTMLElement) {
	for (const child of Array.from(root.childNodes)) {
		if (child.nodeName !== 'BR') continue;

		const isPlaceholder = root.childNodes.length === 1;
		const isLeadingHatch = !child.previousSibling && isCodeBlockElement(child.nextSibling);
		const isTrailingHatch = !child.nextSibling && isCodeBlockElement(child.previousSibling);

		// A hatch goes stale once real content takes over its line:
		// content before a leading hatch, content after a trailing one,
		// or a text node after the block already providing the line.
		// A `<br>` with no code block around is a real newline (browser
		// Shift+Enter shape) and stays.
		let prevElement = child.previousSibling;

		while (prevElement && prevElement.nodeType !== Node.ELEMENT_NODE) {
			prevElement = prevElement.previousSibling;
		}
		const nearBlock =
			isCodeBlockElement(child.nextSibling) ||
			isCodeBlockElement(child.previousSibling) ||
			isCodeBlockElement(prevElement);

		if (!isPlaceholder && !isLeadingHatch && !isTrailingHatch && nearBlock) {
			child.remove();
		}
	}

	for (const child of Array.from(root.childNodes)) {
		if (!isCodeBlockElement(child)) continue;

		if (!hasLineBeside(child.nextSibling)) {
			child.after(document.createElement('br'));
		}
	}
}

/**
 * Strip the separator and artificial newlines from an all-newline text
 * node directly after a fenced block. Chromium's line break at the
 * buffer end inserts an extra artificial `\n` so the new line has
 * height, and the first `\n` after a block doubles as the fence's
 * separator line (synthesized at serialization time). Removing both
 * makes Shift+Enter after a block land the caret on the line directly
 * below the block, like a plain textarea would.
 *
 * Only all-newline text nodes are touched: a node with real content
 * carries intentional blank lines and is left alone. Returns true when
 * the DOM changed.
 */
export function stripBlockBoundaryLineBreaks(root: HTMLElement): boolean {
	let changed = false;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType !== Node.TEXT_NODE) continue;

		if (!isCodeBlockElement(child.previousSibling)) continue;

		let text = child.textContent ?? '';

		if (!/^\n{2,}$/.test(text)) continue;

		text = text.slice(1);

		const atBufferEnd = !child.nextSibling || child.nextSibling.nodeName === 'BR';

		if (atBufferEnd) {
			text = text.slice(0, -1);
		}

		child.textContent = text;
		changed = true;
	}

	return changed;
}

const WORD_CHAR_RE = /[\p{L}\p{N}_]/u;

/**
 * Word-jump target (Option+Arrow / Ctrl+Arrow) in source offsets, or null
 * when the jump crosses no badge and native word movement should handle
 * it. Badge spans are masked to word characters, so a badge counts as
 * exactly one word.
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
		// Entering a badge completes the word phase at the badge's end edge.
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
 * 0 when `caret` sits exactly at a leading badge's end edge, null
 * otherwise. Plain ArrowLeft there has no native previous position, so
 * the host snaps the caret to the buffer start manually.
 */
export function leadingBadgeEdgeOffset(source: string, caret: number): number | null {
	const [first] = tokenizeContent(source);

	if (!first || first.kind !== 'badge') return null;

	return caret === badgeSourceLength(first.name, first.path) ? 0 : null;
}

/**
 * Translate a plain-text offset into a degenerate `Range` at that
 * position in the DOM; out-of-range offsets clamp to buffer end (before
 * a trailing escape hatch, not after it). Zero offset lands BEFORE a
 * badge or code span, and an offset exactly at a code span's end lands
 * AFTER it, so typing at a code span's edge extends the surrounding
 * text. Interior code-span offsets land in the element's text.
 * Understands the same block/`<br>` newline shapes as
 * `serializeContent`.
 */
export function textOffsetToRange(root: HTMLElement, offset: number): Range {
	const range = document.createRange();

	let remaining = offset;
	let landed = false;
	let pendingBlockBoundary = false;

	const land = (node: Node, nodeOffset: number) => {
		range.setStart(node, nodeOffset);
		range.setEnd(node, nodeOffset);
		landed = true;
	};
	const walk = (parent: Node) => {
		let first = true;

		for (const child of Array.from(parent.childNodes)) {
			if (landed) return;

			if (child.nodeType === Node.TEXT_NODE) {
				const text = child.textContent ?? '';

				if (text.length === 0) continue;

				if (pendingBlockBoundary) {
					// The synthesized separator maps to the near edge of the
					// content that follows the block.
					pendingBlockBoundary = false;

					if (remaining === 0) {
						land(child, 0);

						return;
					}

					remaining -= 1;
				}

				if (remaining <= text.length) {
					land(child, remaining);

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

				if (pendingBlockBoundary) {
					pendingBlockBoundary = false;

					if (remaining === 0) {
						range.setStartBefore(el);
						range.setEndBefore(el);
						landed = true;

						return;
					}

					remaining -= 1;
				}

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

			if (el.dataset.codeToken !== undefined) {
				const isBlock = el.dataset.codeToken === 'block';

				if (isBlock && (pendingBlockBoundary || !first)) {
					pendingBlockBoundary = false;

					if (remaining === 0) {
						range.setStartBefore(el);
						range.setEndBefore(el);
						landed = true;

						return;
					}

					remaining -= 1;
				}

				const len = (el.textContent ?? '').length;

				if (remaining === 0) {
					range.setStartBefore(el);
					range.setEndBefore(el);
					landed = true;

					return;
				}

				if (remaining === len) {
					range.setStartAfter(el);
					range.setEndAfter(el);
					landed = true;

					return;
				}

				if (remaining < len) {
					walk(el);

					return;
				}

				remaining -= len;

				if (isBlock) remaining -= 1;

				first = false;

				continue;
			}

			if (el.tagName === 'BR') {
				const isHatch =
					isCodeBlockElement(el.previousSibling) || isCodeBlockElement(el.nextSibling);

				if (isHatch) {
					// Escape hatch: no source length; offset 0 lands before it
					// so text typed there takes its place.
					if (remaining === 0) {
						range.setStartBefore(el);
						range.setEndBefore(el);
						landed = true;
					}

					continue;
				}

				if (!el.nextSibling) continue;

				if (pendingBlockBoundary) {
					pendingBlockBoundary = false;

					if (remaining === 0) {
						range.setStartBefore(el);
						range.setEndBefore(el);
						landed = true;

						return;
					}

					remaining -= 1;
				}

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
				if (pendingBlockBoundary || !first) {
					pendingBlockBoundary = false;

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
		const last = root.lastChild;

		if (last && last.nodeName === 'BR') {
			range.setStartBefore(last);
			range.setEndBefore(last);
		} else {
			range.selectNodeContents(root);
			range.collapse(false);
		}
	}

	return range;
}
