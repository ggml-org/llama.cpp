// Questions about the markdown under a position, answered from the editor's syntax
// tree, so they agree with what the live preview renders (``` and ~~~ fences,
// indented code, inline code).

import { ensureSyntaxTree, syntaxTree } from '@codemirror/language';
import type { EditorState } from '@codemirror/state';
import type { SyntaxNode } from '@lezer/common';

const CODE_NODE_NAMES = new Set(['CodeBlock', 'FencedCode', 'InlineCode']);
const PARSE_TIMEOUT_MS = 50;

/**
 * Is `pos` inside a code block, fenced (closed, or still open while it is being
 * typed) or indented? Right after a closing fence counts as outside.
 */
export function isInCodeBlock(state: EditorState, pos: number): boolean {
	const tree = ensureSyntaxTree(state, pos, PARSE_TIMEOUT_MS) ?? syntaxTree(state);

	for (let node: SyntaxNode | null = tree.resolveInner(pos, -1); node; node = node.parent) {
		if (node.name === 'CodeBlock') return true;

		if (node.name === 'FencedCode') {
			const closing = node.lastChild;
			const closed = closing?.name === 'CodeMark' && closing.from > node.from;

			return !(closed && pos >= node.to);
		}
	}

	return false;
}

/** The [from, to) ranges of code (blocks and inline) between `from` and `to`. */
export function codeRanges(state: EditorState, from: number, to: number): [number, number][] {
	const ranges: [number, number][] = [];

	syntaxTree(state).iterate({
		enter: (node) => {
			if (!CODE_NODE_NAMES.has(node.name)) return;

			ranges.push([node.from, node.to]);

			return false;
		},
		from,
		to
	});

	return ranges;
}
