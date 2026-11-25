import { rehypeLazyKatex } from '#lib/components/app/content/MarkdownContent/plugins/rehype/lazy-katex.ts';
import type { Root } from 'hast';
import rehypeStringify from 'rehype-stringify';
import { remark } from 'remark';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import { describe, expect, it } from 'vitest';

const processor = remark()
	.use(remarkMath)
	.use(remarkRehype)
	.use(rehypeLazyKatex)
	.use(rehypeStringify);

async function render(markdown: string): Promise<string> {
	return processor.stringify((await processor.run(processor.parse(markdown))) as Root);
}

describe('rehypeLazyKatex', () => {
	it('renders inline and display math with KaTeX', async () => {
		const html = await render('Area $\\pi r^2$\n\n$$\nE = mc^2\n$$');

		expect(html.match(/class="katex"/g)).toHaveLength(2);
		expect(html).toContain('katex-display');
	});

	it('leaves a message without math as it is, in the same pass', () => {
		const tree = processor.runSync(processor.parse('Just *text* and `code`')) as Root;

		expect(processor.stringify(tree)).toBe('<p>Just <em>text</em> and <code>code</code></p>');
	});
});
