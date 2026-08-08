// Safety net for MarkdownContent's incremental rendering.
//
// The component renders streamed markdown by caching already-settled blocks and
// only re-rendering the trailing one. Whatever it does internally, the result
// after streaming a document in must be identical to rendering that same
// document in one shot. These tests pin that invariant so the incremental
// machinery can be changed with confidence.

import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import MarkdownStreamWrapper from './components/MarkdownStreamWrapper.svelte';
import { markdownState } from './components/markdown-stream-state.svelte';

function nextFrame(): Promise<void> {
	return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/** Let MarkdownContent's rAF-coalesced processing drain. */
async function settle() {
	for (let i = 0; i < 4; i++) {
		await tick();
		await nextFrame();
	}
	await tick();
}

/**
 * Code blocks get ids from a global counter that advances every time one is
 * rendered, so a streamed document (which renders its trailing block repeatedly)
 * legitimately lands on higher numbers than a one-shot render. Normalise them so
 * the comparison is about structure and content, not render count.
 */
function normalize(html: string): string {
	return html.replace(/code-\d+/g, 'code-N');
}

function host(): HTMLElement {
	const el = document.querySelector('[data-testid="markdown-host"]');
	if (!el) throw new Error('markdown host not mounted');
	return el as HTMLElement;
}

/** Render `doc` in one shot and return the resulting markup. */
async function renderWhole(doc: string): Promise<string> {
	markdownState.content = '';
	const { unmount } = render(MarkdownStreamWrapper);
	await settle();

	markdownState.content = doc;
	await settle();

	const html = normalize(host().innerHTML);
	await unmount();

	return html;
}

/** Stream `doc` in `chunks` pieces and return the resulting markup. */
async function renderStreamed(doc: string, chunks: number): Promise<string> {
	markdownState.content = '';
	const { unmount } = render(MarkdownStreamWrapper);
	await settle();

	const size = Math.ceil(doc.length / chunks);
	for (let i = 0; i < doc.length; i += size) {
		markdownState.content = doc.slice(0, i + size);
		await settle();
	}

	markdownState.content = doc;
	await settle();

	const html = normalize(host().innerHTML);
	await unmount();

	return html;
}

const DOCS: Array<[string, string]> = [
	['plain paragraphs', 'First paragraph here.\n\nSecond paragraph.\n\nThird one to finish.'],
	[
		'headings and lists',
		'# Title\n\nIntro text.\n\n## Section\n\n- alpha\n- beta\n- gamma\n\n1. one\n2. two\n\nClosing words.'
	],
	[
		'fenced code between prose',
		'Before the block.\n\n```ts\nconst x: number = 1;\nconsole.log(x);\n```\n\nAfter the block.'
	],
	[
		'table then prose',
		'Intro.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n\nTrailing paragraph.'
	],
	[
		'blockquote and emphasis',
		'> quoted line\n> continues here\n\nSome **bold** and _italic_ text.'
	],
	['single long paragraph', `A single unbroken paragraph. ${'More words follow. '.repeat(40)}`],
	[
		'inline math and currency',
		'Cost is $5 today.\n\nThe identity $e^{i\\pi} + 1 = 0$ is famous.\n\nDone.'
	],
	[
		'nested list with code',
		'- outer\n  - inner one\n  - inner two\n\n```js\nlet y = 2;\n```\n\nEnd.'
	]
];

// Constructs whose meaning depends on a line that arrives *later*, so a naive
// incremental parser that commits at every newline would settle them wrongly:
// the setext underline retitles the line above it, the table separator turns the
// header row into a table, and lazy continuation glues lines into one paragraph.
const BOUNDARY_SENSITIVE_DOCS: Array<[string, string]> = [
	['setext heading', 'Title\n=====\n\nBody.'],
	['setext h2', 'Sub\n---\n\nBody.'],
	['table needs its separator', '| a | b |\n| --- | --- |\n| 1 | 2 |\n\nAfter.'],
	['lazy continuation', 'One line\ncontinued lazily.\n\nNext para.'],
	['blockquote lazy continuation', '> quoted\nstill quoted\n\nOut.'],
	['list item continuation', '- item\n  continued\n\nAfter.'],
	// The hard case: a definition at the END of the document changes how an
	// EARLIER paragraph renders, so committing that paragraph early is wrong.
	['link reference definition', 'See [foo] here.\n\nMore text.\n\n[foo]: /url\n'],
	// Prose, then a fence that CLOSES mid-stream, then more prose. Exercises the
	// hand-off from the incomplete-code-block branch back to the standard path,
	// where reused blocks must re-acquire an endOffset for the commit loop.
	[
		'prose then closed fence then prose',
		'Intro paragraph before any code.\n\n```ts\nconst a = 1;\nconst b = a + 1;\n```\n\nMiddle prose after the fence.\n\n```js\nlet c = 3;\n```\n\nClosing paragraph.'
	]
];

describe('MarkdownContent streaming matches one-shot rendering', () => {
	for (const [label, doc] of DOCS) {
		it(`${label}: streamed in 5 chunks equals whole`, { timeout: 60_000 }, async () => {
			const whole = await renderWhole(doc);
			const streamed = await renderStreamed(doc, 5);

			if (streamed !== whole) {
				// Print a compact first-difference report instead of two 10KB blobs.
				let i = 0;
				while (i < Math.min(streamed.length, whole.length) && streamed[i] === whole[i]) i++;
				console.log(
					`\n[${label}] first diff at ${i}\n` +
						`  whole:    ...${JSON.stringify(whole.slice(Math.max(0, i - 60), i + 120))}\n` +
						`  streamed: ...${JSON.stringify(streamed.slice(Math.max(0, i - 60), i + 120))}`
				);
			}

			expect(streamed).toBe(whole);
		});
	}

	it('token-by-token streaming equals whole', { timeout: 120_000 }, async () => {
		const doc = DOCS[1][1];
		const whole = await renderWhole(doc);
		const streamed = await renderStreamed(doc, doc.length);

		expect(streamed).toBe(whole);
	});

	// These are the cases that decide where a streaming parser may safely stop
	// re-parsing. Streamed character by character, so every intermediate state is
	// exercised - including the one just before the disambiguating line arrives.
	for (const [label, doc] of BOUNDARY_SENSITIVE_DOCS) {
		it(`${label}: token-by-token equals whole`, { timeout: 120_000 }, async () => {
			const whole = await renderWhole(doc);
			const streamed = await renderStreamed(doc, doc.length);

			expect(streamed).toBe(whole);
		});
	}
});
