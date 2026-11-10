// Where the code is in the markdown, answered from the editor's syntax tree, so it agrees
// with what the live preview renders (``` and ~~~ fences, indented code, inline code).

import { syntaxTree } from '@codemirror/language';
import type { EditorState } from '@codemirror/state';

const CODE_NODE_NAMES = new Set(['CodeBlock', 'FencedCode', 'InlineCode']);

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
