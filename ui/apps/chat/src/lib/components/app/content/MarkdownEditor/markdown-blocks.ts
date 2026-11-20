// The top-level markdown blocks of the editor's document, read from its syntax tree:
// the units the live preview either renders or shows as raw text.

import { syntaxTree, syntaxTreeAvailable } from '@codemirror/language';
import type { EditorState } from '@codemirror/state';

export interface MarkdownBlock {
	from: number;
	/** End of the block's own text. */
	sourceTo: number;
	/** End of the block's range, including the blank lines that follow it. */
	to: number;
	name: string;
	/** The collapsed margin between this block and the one before it, in rem. */
	gapRem: number;
	/** The block's own text, from..sourceTo. */
	source: string;
}

// [top, bottom] margins in rem of each block kind as a chat message renders it (each
// block in a .markdown-block of its own, so a heading is a first child). CodeMirror
// can not measure margins on block widgets, so the collapsed gap between two blocks is
// computed here and applied as padding. markdown-editor-margins.svelte.test.ts checks
// them against the stylesheet.
export const BLOCK_MARGINS_REM: Record<string, [number, number]> = {
	ATXHeading1: [0.5, 0.75],
	ATXHeading2: [0.5, 0.5],
	ATXHeading3: [0.5, 0.5],
	ATXHeading4: [0.5, 0.25],
	ATXHeading5: [0.5, 0.25],
	ATXHeading6: [0.5, 0.25],
	Blockquote: [1.5, 1.5],
	BulletList: [1, 1],
	CodeBlock: [1, 1],
	FencedCode: [1, 1],
	HorizontalRule: [2, 2],
	OrderedList: [1, 1],
	SetextHeading1: [0.5, 0.75],
	SetextHeading2: [0.5, 0.5],
	Table: [1, 1]
};
export const PARAGRAPH_MARGINS_REM: [number, number] = [1, 1];
export const CODE_BLOCK_NAMES = new Set(['CodeBlock', 'FencedCode']);
const MATH_FENCE = '$$';

function blockMargins(name: string): [number, number] {
	return BLOCK_MARGINS_REM[name] ?? PARAGRAPH_MARGINS_REM;
}

/** How many lines of `text` open or close a `$$` math block (`$$x$$` on one line does neither). */
function mathFenceCount(text: string): number {
	if (!text.includes(MATH_FENCE)) return 0;

	let count = 0;

	for (const line of text.split('\n')) {
		const trimmed = line.trim();

		if (!trimmed.startsWith(MATH_FENCE)) continue;

		if (trimmed.length > 2 * MATH_FENCE.length && trimmed.endsWith(MATH_FENCE)) continue;

		count++;
	}

	return count;
}

/**
 * A `$$` math block may hold blank lines, which split it into several top-level nodes
 * for the markdown parser; join them back into one block, as the renderer sees it.
 */
function joinMathBlocks(
	blocks: MarkdownBlock[],
	join: (first: MarkdownBlock, last: MarkdownBlock) => MarkdownBlock
): MarkdownBlock[] {
	const opens = (block: MarkdownBlock) =>
		!CODE_BLOCK_NAMES.has(block.name) && mathFenceCount(block.source) % 2 === 1;
	const joined: MarkdownBlock[] = [];

	for (let index = 0; index < blocks.length; index++) {
		const block = blocks[index];

		if (!opens(block)) {
			joined.push(block);

			continue;
		}

		let close = index + 1;

		while (close < blocks.length && !opens(blocks[close])) close++;

		// never closed: the block stays as the parser cut it
		if (close === blocks.length) {
			joined.push(block);

			continue;
		}

		joined.push(join(block, blocks[close]));
		index = close;
	}

	return joined;
}

export function collectBlocks(state: EditorState): MarkdownBlock[] {
	const { doc } = state;
	// A long document may not be parsed to its end yet: its last parsed node may be cut
	// short, so it and the rest stay raw until the parser catches up (the tree change
	// collects the blocks again)
	const complete = syntaxTreeAvailable(state, doc.length);
	// `to` and `gapRem` depend on the neighbours, so they are set once the blocks are known
	const blockOf = (name: string, from: number, sourceTo: number): MarkdownBlock => ({
		from,
		gapRem: 0,
		name,
		source: doc.sliceString(from, sourceTo),
		sourceTo,
		to: sourceTo
	});

	let blocks: MarkdownBlock[] = [];

	for (let node = syntaxTree(state).topNode.firstChild; node; node = node.nextSibling) {
		blocks.push(blockOf(node.name, doc.lineAt(node.from).from, doc.lineAt(node.to).to));
	}

	if (!complete) blocks.pop();

	blocks = joinMathBlocks(blocks, (first, last) => blockOf(first.name, first.from, last.sourceTo));

	blocks.forEach((block, index) => {
		const next = blocks[index + 1];
		const prev = blocks[index - 1];

		if (next) block.to = next.from - 1;
		else if (complete) block.to = doc.length;

		if (prev) block.gapRem = Math.max(blockMargins(prev.name)[1], blockMargins(block.name)[0]);
	});

	return blocks;
}
