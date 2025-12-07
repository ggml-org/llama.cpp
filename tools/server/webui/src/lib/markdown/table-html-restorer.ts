/**
 * Rehype plugin to restore limited HTML elements inside Markdown table cells.
 *
 * ## Problem
 * The remark/rehype pipeline neutralizes inline HTML as literal text
 * (remarkLiteralHtml) so that XML/HTML snippets in LLM responses display
 * as-is instead of being rendered. This causes <br> and <ul> markup in
 * table cells to show as plain text.
 *
 * ## Solution
 * This plugin traverses the HAST post-conversion, parses whitelisted HTML
 * patterns from text nodes, and replaces them with actual HAST element nodes
 * that will be rendered as real HTML.
 *
 * ## Supported HTML
 * - `<br>` / `<br/>` / `<br />` - Line breaks (inline)
 * - `<ul><li>...</li></ul>` - Unordered lists (block)
 *
 * ## Key Implementation Details
 *
 * ### 1. Sibling Combination (Critical)
 * The Markdown pipeline may fragment content across multiple text nodes and `<br>`
 * elements. For example, `<ul><li>a</li></ul>` might arrive as:
 *   - Text: `"<ul>"`
 *   - Element: `<br>`
 *   - Text: `"<li>a</li></ul>"`
 *
 * We must combine consecutive text nodes and `<br>` elements into a single string
 * before attempting to parse list markup. Without this, list detection fails.
 *
 * ### 2. visitParents for Deep Traversal
 * Table cell content may be wrapped in intermediate elements (e.g., `<p>` tags).
 * Using `visitParents` instead of direct child iteration ensures we find text
 * nodes at any depth within the cell.
 *
 * ### 3. Reference Comparison for No-Op Detection
 * When checking if `<br>` expansion changed anything, we compare:
 *   `expanded.length !== 1 || expanded[0] !== textNode`
 *
 * This catches both cases:
 * - Multiple nodes created (text was split)
 * - Single NEW node created (original had only `<br>`, now it's an element)
 *
 * A simple `length > 1` check would miss the single `<br>` case.
 *
 * ### 4. Strict List Validation
 * `parseList()` rejects malformed markup by checking for garbage text between
 * `<li>` elements. This prevents creating broken DOM from partial matches like
 * `<ul>garbage<li>a</li></ul>`.
 *
 * ### 5. Newline Substitution for `<br>` in Combined String
 * When combining siblings, existing `<br>` elements become `\n` in the combined
 * string. This allows list content to span visual lines while still being parsed
 * as a single unit.
 *
 * @example
 * // Input Markdown:
 * // | Feature | Notes |
 * // |---------|-------|
 * // | Multi-line | First<br>Second |
 * // | List | <ul><li>A</li><li>B</li></ul> |
 * //
 * // Without this plugin: <br> and <ul> render as literal text
 * // With this plugin: <br> becomes line break, <ul> becomes actual list
 */

import type { Plugin } from 'unified';
import type { Element, ElementContent, Root, Text } from 'hast';
import { visit } from 'unist-util-visit';
import { visitParents } from 'unist-util-visit-parents';
import { BR_PATTERN, LIST_PATTERN, LI_PATTERN } from '$lib/constants/table-html-restorer';

/**
 * Expands text containing `<br>` tags into an array of text nodes and br elements.
 */
function expandBrTags(value: string): ElementContent[] {
	const matches = [...value.matchAll(BR_PATTERN)];
	if (!matches.length) return [{ type: 'text', value } as Text];

	const result: ElementContent[] = [];
	let cursor = 0;

	for (const m of matches) {
		if (m.index! > cursor) {
			result.push({ type: 'text', value: value.slice(cursor, m.index) } as Text);
		}
		result.push({ type: 'element', tagName: 'br', properties: {}, children: [] } as Element);
		cursor = m.index! + m[0].length;
	}

	if (cursor < value.length) {
		result.push({ type: 'text', value: value.slice(cursor) } as Text);
	}

	return result;
}

/**
 * Parses a `<ul><li>...</li></ul>` string into a HAST element.
 * Returns null if the markup is malformed or contains unexpected content.
 */
function parseList(value: string): Element | null {
	const match = value.trim().match(LIST_PATTERN);
	if (!match) return null;

	const body = match[1];
	const items: ElementContent[] = [];
	let cursor = 0;

	for (const liMatch of body.matchAll(LI_PATTERN)) {
		// Reject if there's non-whitespace between list items
		if (body.slice(cursor, liMatch.index!).trim()) return null;

		items.push({
			type: 'element',
			tagName: 'li',
			properties: {},
			children: expandBrTags(liMatch[1] ?? '')
		} as Element);

		cursor = liMatch.index! + liMatch[0].length;
	}

	// Reject if no items found or trailing garbage exists
	if (!items.length || body.slice(cursor).trim()) return null;

	return { type: 'element', tagName: 'ul', properties: {}, children: items } as Element;
}

/**
 * Processes a single table cell, restoring HTML elements from text content.
 */
function processCell(cell: Element) {
	visitParents(cell, 'text', (textNode: Text, ancestors) => {
		const parent = ancestors[ancestors.length - 1];
		if (!parent || parent.type !== 'element') return;

		const parentEl = parent as Element;
		const siblings = parentEl.children as ElementContent[];
		const startIndex = siblings.indexOf(textNode as ElementContent);
		if (startIndex === -1) return;

		// Combine consecutive text nodes and <br> elements into one string
		let combined = '';
		let endIndex = startIndex;

		for (let i = startIndex; i < siblings.length; i++) {
			const sib = siblings[i];
			if (sib.type === 'text') {
				combined += (sib as Text).value;
				endIndex = i;
			} else if (sib.type === 'element' && (sib as Element).tagName === 'br') {
				combined += '\n';
				endIndex = i;
			} else {
				break;
			}
		}

		// Try parsing as list first (replaces entire combined range)
		const list = parseList(combined);
		if (list) {
			siblings.splice(startIndex, endIndex - startIndex + 1, list);
			return;
		}

		// Otherwise, just expand <br> tags in this text node
		const expanded = expandBrTags(textNode.value);
		if (expanded.length !== 1 || expanded[0] !== textNode) {
			siblings.splice(startIndex, 1, ...expanded);
		}
	});
}

export const rehypeRestoreTableHtml: Plugin<[], Root> = () => (tree) => {
	visit(tree, 'element', (node: Element) => {
		if (node.tagName === 'td' || node.tagName === 'th') {
			processCell(node);
		}
	});
};
