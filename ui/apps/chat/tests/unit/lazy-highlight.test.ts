import { rehypeLazyHighlight } from '#lib/components/app/content/MarkdownContent/plugins/rehype/lazy-highlight.ts';
import {
	highlightCode,
	isHighlightLanguageLoaded,
	loadHighlightLanguages,
	resolveHighlightLanguage
} from '#lib/utils/code.ts';
import type { Root } from 'hast';
import rehypeStringify from 'rehype-stringify';
import { remark } from 'remark';
import remarkRehype from 'remark-rehype';
import { describe, expect, it } from 'vitest';

// one file, so these run in order against one fresh loader: nothing has loaded when it starts
describe('highlightCode before its language loads', () => {
	it('resolves aliases to highlight.js names', () => {
		expect(resolveHighlightLanguage('ts')).toBe('typescript');
		expect(resolveHighlightLanguage('SH')).toBe('bash');
		expect(resolveHighlightLanguage('yml')).toBe('yaml');
		expect(resolveHighlightLanguage('python')).toBe('python');
		expect(resolveHighlightLanguage('constructor')).toBeUndefined();
		expect(resolveHighlightLanguage('not-a-language')).toBeUndefined();
	});

	it('returns plain text, then highlights once that language (and only it) has loaded', async () => {
		expect(highlightCode('\nconst x = 1;\n', 'ts')).toBe('const x = 1;');

		await loadHighlightLanguages(['ts']);

		expect(isHighlightLanguageLoaded('typescript')).toBe(true);
		expect(highlightCode('const x = 1;', 'ts')).toContain('hljs-keyword');
		expect(isHighlightLanguageLoaded('python')).toBe(false);
	});

	it('loads the languages a language embeds with it', async () => {
		await loadHighlightLanguages(['html']);

		expect(isHighlightLanguageLoaded('css')).toBe(true);
		expect(highlightCode('<style>a { color: red; }</style>', 'html')).toContain('hljs-attribute');
	});

	it('leaves an unknown language plain without auto-detection', () => {
		expect(highlightCode('<b>a && b</b>', 'not-a-language', false)).toBe(
			'&lt;b&gt;a &amp;&amp; b&lt;/b&gt;'
		);
	});

	it('auto-detects an unknown language among the common languages once they load', async () => {
		const code = 'def greet(name):\n    return f"hi {name}"';

		expect(highlightCode(code, 'not-a-language')).toBe(code);

		// that call started loading every common language
		const html = await waitForHighlight(() => highlightCode(code, 'not-a-language'));

		expect(html).toContain('hljs-');
	});
});

async function waitForHighlight(highlight: () => string): Promise<string> {
	for (let i = 0; i < 50; i++) {
		const html = highlight();

		if (html.includes('hljs-')) return html;

		await new Promise((resolve) => setTimeout(resolve, 20));
	}

	return highlight();
}

const processor = remark()
	.use(remarkRehype)
	.use(rehypeLazyHighlight, { aliases: { xml: ['svelte'] } })
	.use(rehypeStringify);

async function render(markdown: string): Promise<string> {
	return processor.stringify((await processor.run(processor.parse(markdown))) as Root);
}

describe('rehypeLazyHighlight', () => {
	it('leaves a message without a code block as it is, in the same pass', () => {
		const tree = processor.runSync(processor.parse('Just *text* and `code`')) as Root;

		expect(processor.stringify(tree)).toBe('<p>Just <em>text</em> and <code>code</code></p>');
	});

	it('loads a block language, then highlights it in the same pass', async () => {
		const markdown = '```rust\nfn main() {}\n```';
		const first = await render(markdown);

		expect(first).toContain('<code class="hljs language-rust">');
		expect(first).toContain('hljs-keyword');

		const second = processor.stringify(processor.runSync(processor.parse(markdown)) as Root);

		expect(second).toBe(first);
	});

	it('uses highlight.js definitions, not highlightCode bash changes', async () => {
		const html = await render('```sh\ncurl -LsSf https://example.com\n```');

		expect(html).toContain('<code class="hljs language-sh">');
		expect(html).not.toContain('hljs-built_in">curl');
	});

	it('takes extra aliases', async () => {
		expect(await render('```svelte\n<div>{x}</div>\n```')).toContain('hljs-tag');
	});

	it('leaves unknown languages, no-highlight and blocks without a language plain', async () => {
		expect(await render('```not-a-language\nx = 1\n```')).toBe(
			'<pre><code class="hljs language-not-a-language">x = 1\n</code></pre>'
		);
		expect(await render('```\nx = 1\n```')).toBe('<pre><code>x = 1\n</code></pre>');
	});
});
