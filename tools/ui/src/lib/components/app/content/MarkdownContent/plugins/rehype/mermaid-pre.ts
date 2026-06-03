import type { Plugin } from 'unified';
import type { Root, Element, ElementContent, Text } from 'hast';
import { visit } from 'unist-util-visit';

/**
 * Recursively extracts all text content from a HAST node.
 * Handles nested elements (e.g., span wrappers from syntax highlighting).
 */
function extractText(node: ElementContent): string {
	if (node.type === 'text') return node.value;
	if (node.type === 'element') {
		return (node.children ?? []).map(extractText).join('');
	}
	return '';
}

/**
 * Rehype plugin to convert mermaid code blocks to <pre class="mermaid"> elements.
 *
 * Transforms:
 *   <pre><code class="language-mermaid">graph TD; A-->B</code></pre>
 * into:
 *   <pre class="mermaid">graph TD; A-->B</pre>
 *
 * The mermaid library renders these client-side via mermaid.run().
 *
 * Must run BEFORE rehypeEnhanceCodeBlocks so mermaid blocks are not wrapped
 * with code block headers/buttons (they have no <code> child, so they're skipped).
 */
export const rehypeMermaidPre: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'pre' || !parent || index === undefined) return;

			const codeElement = node.children.find(
				(child): child is Element => child.type === 'element' && child.tagName === 'code'
			);

			if (!codeElement) return;

			const className = codeElement.properties?.className;
			if (!Array.isArray(className)) return;

			const isMermaid = className.some(
				(cls) => typeof cls === 'string' && cls === 'language-mermaid'
			);

			if (!isMermaid) return;

			// Recursively extract text to handle nested spans from syntax highlighting
			const diagramText = codeElement.children.map(extractText).join('').trim();

			if (!diagramText) return;

			const mermaidPre: Element = {
				type: 'element',
				tagName: 'pre',
				properties: {
					className: ['mermaid']
				},
				children: [{ type: 'text', value: diagramText } as Text]
			};

			(parent.children as ElementContent[])[index] = mermaidPre;
		});
	};
};
