import { markdownLanguage } from '@codemirror/lang-markdown';

/**
 * Does `source` hold any markdown block other than plain paragraphs (a heading,
 * list, quote, code block, table, rule, ...)? Inline markup inside a paragraph
 * does not count.
 */
export function hasMarkdownStructure(source: string): boolean {
	const tree = markdownLanguage.parser.parse(source);

	for (let node = tree.topNode.firstChild; node; node = node.nextSibling) {
		if (node.name !== 'Paragraph') return true;
	}

	return false;
}
