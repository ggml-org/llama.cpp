// The live preview spaces its rendered blocks from a table of block margins, as
// CodeMirror can't measure the margins of block widgets. This measures what a chat
// message actually shows for each block kind, so the table can't drift from
// markdown-content.css.

import MarkdownContent from '$lib/components/app/content/MarkdownContent/MarkdownContent.svelte';
import {
	BLOCK_MARGINS_REM,
	PARAGRAPH_MARGINS_REM
} from '$lib/components/app/content/MarkdownEditor/markdown-blocks';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

// Each kind sits between two paragraphs, as most blocks do
const SAMPLES: Record<string, { markdown: string }> = {
	ATXHeading1: { markdown: '# x' },
	ATXHeading2: { markdown: '## x' },
	ATXHeading3: { markdown: '### x' },
	ATXHeading4: { markdown: '#### x' },
	ATXHeading5: { markdown: '##### x' },
	ATXHeading6: { markdown: '###### x' },
	Blockquote: { markdown: '> x' },
	BulletList: { markdown: '- x' },
	CodeBlock: { markdown: '    x' },
	FencedCode: { markdown: '```\nx\n```' },
	HorizontalRule: { markdown: '***' },
	OrderedList: { markdown: '1. x' },
	Paragraph: { markdown: 'x' },
	SetextHeading1: { markdown: 'x\n===' },
	SetextHeading2: { markdown: 'x\n---' },
	Table: { markdown: '| a |\n| - |\n| b |' }
};

function rem(px: string): number {
	return parseFloat(px) / parseFloat(getComputedStyle(document.documentElement).fontSize);
}

describe('live preview block margins', () => {
	for (const [name, sample] of Object.entries(SAMPLES)) {
		it(`match a chat message for ${name}`, async () => {
			const screen = render(MarkdownContent, { content: `before\n\n${sample.markdown}\n\nafter` });
			const blocks = () => screen.container.querySelectorAll('.markdown-block');

			await vi.waitFor(() => expect(blocks().length).toBe(3));

			// the block's outer elements carry the margins that space it from its neighbours
			const block = blocks()[1];
			const first = block.firstElementChild;
			const last = block.lastElementChild;

			if (!first || !last) throw new Error(`${name} not rendered`);

			const expected = BLOCK_MARGINS_REM[name] ?? PARAGRAPH_MARGINS_REM;

			expect([
				rem(getComputedStyle(first).marginTop),
				rem(getComputedStyle(last).marginBottom)
			]).toEqual(expected);
		});
	}
});
