import type { Plugin } from 'unified';
import type { Root, Element, ElementContent, Text } from 'hast';
import { visit } from 'unist-util-visit';

const MAXIMIZE_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

function createIconElement(svg: string): Element {
	return {
		type: 'element',
		tagName: 'span',
		properties: {},
		children: [{ type: 'raw', value: svg } as unknown as ElementContent]
	};
}

/**
 * Rehype plugin to convert mermaid code blocks to wrapped elements with fullscreen button.
 *
 * Transforms:
 *   <pre><code class="language-mermaid">graph TD; A-->B</code></pre>
 * into:
 *   <div class="mermaid-wrapper">
 *     <pre class="mermaid">graph TD; A-->B</pre>
 *     <button class="mermaid-fullscreen-btn" data-mermaid-code="graph TD; A-->B" title="Expand diagram">
 *       (maximize icon)
 *     </button>
 *   </div>
 *
 * The mermaid library renders the <pre class="mermaid"> client-side via mermaid.run().
 * The fullscreen button stores the diagram source for the DialogMermaidPreview component.
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

			const fullscreenButton: Element = {
				type: 'element',
				tagName: 'button',
				properties: {
					className: ['mermaid-fullscreen-btn'],
					title: 'Expand diagram',
					type: 'button',
					'data-mermaid-code': diagramText
				},
				children: [createIconElement(MAXIMIZE_ICON_SVG)]
			};

			const wrapper: Element = {
				type: 'element',
				tagName: 'div',
				properties: {
					className: ['mermaid-wrapper']
				},
				children: [mermaidPre, fullscreenButton]
			};

			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
