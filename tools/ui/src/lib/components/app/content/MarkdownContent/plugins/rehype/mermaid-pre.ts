import type { Plugin } from 'unified';
import type { Root, Element, ElementContent, Text } from 'hast';
import { visit } from 'unist-util-visit';

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

			const diagramText = codeElement.children
				.map((child) => {
					if (child.type === 'text') return child.value;
					return '';
				})
				.join('')
				.trim();

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
