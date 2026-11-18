import { rehypeLazyHighlight } from '#lib/components/app/content/MarkdownContent/plugins/rehype/lazy-highlight.ts';
import { highlightCode, isHighlighterLoaded, loadHighlighter } from '#lib/utils/code.ts';
import type { Root } from 'hast';
import rehypeStringify from 'rehype-stringify';
import { remark } from 'remark';
import remarkRehype from 'remark-rehype';
import { describe, expect, it } from 'vitest';

describe('highlightCode before highlight.js loads', () => {
	it('returns the code as escaped plain text and starts loading highlight.js', async () => {
		expect(isHighlighterLoaded()).toBe(false);
		expect(highlightCode('\n<b>a && b</b>\n', 'html')).toBe('&lt;b&gt;a &amp;&amp; b&lt;/b&gt;');

		await loadHighlighter();

		expect(isHighlighterLoaded()).toBe(true);
		expect(highlightCode('<b>a && b</b>', 'html')).toContain('hljs-tag');
	});
});

const processor = remark()
	.use(remarkRehype)
	.use(rehypeLazyHighlight, { aliases: { xml: ['svelte'] } })
	.use(rehypeStringify);

describe('rehypeLazyHighlight', () => {
	it('leaves a message without a code block as it is, in the same pass', () => {
		const tree = processor.runSync(processor.parse('Just *text* and `code`')) as Root;

		expect(processor.stringify(tree)).toBe('<p>Just <em>text</em> and <code>code</code></p>');
	});

	it('highlights code blocks once highlight.js has loaded, then in the same pass', async () => {
		const markdown = '```ts\nconst answer = 42;\n```';
		const first = processor.stringify((await processor.run(processor.parse(markdown))) as Root);

		expect(first).toContain('<code class="hljs language-ts">');
		expect(first).toContain('hljs-keyword');

		const second = processor.stringify(processor.runSync(processor.parse(markdown)) as Root);

		expect(second).toBe(first);
	});

	it('passes its options to rehype-highlight', async () => {
		const html = processor.stringify(
			(await processor.run(processor.parse('```svelte\n<div>{x}</div>\n```'))) as Root
		);

		expect(html).toContain('hljs-tag');
	});
});
