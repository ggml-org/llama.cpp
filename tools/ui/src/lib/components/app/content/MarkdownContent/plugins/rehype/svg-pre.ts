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
 * Rehype plugin to convert svg code blocks to <pre class="svg-diagram"> elements.
 *
 * Transforms:
 *   <pre><code class="language-svg"><svg>...</svg></code></pre>
 * into:
 *   <pre class="svg-diagram"><svg>...</svg></pre>
 *
 * The raw svg text is carried as a text child. Rendering and sanitization
 * happen client-side, so no markup is injected at this stage.
 *
 * Must run BEFORE rehypeEnhanceCodeBlocks so svg blocks are not wrapped
 * with code block headers/buttons (they have no <code> child, so they're skipped).
 */
export const rehypeSvgPre: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'pre' || !parent || index === undefined) return;

			const codeElement = node.children.find(
				(child): child is Element => child.type === 'element' && child.tagName === 'code'
			);

			if (!codeElement) return;

			const className = codeElement.properties?.className;
			if (!Array.isArray(className)) return;

			const isSvg = className.some((cls) => typeof cls === 'string' && cls === 'language-svg');

			if (!isSvg) return;

			// Recursively extract text to handle nested spans from syntax highlighting
			const svgText = codeElement.children.map(extractText).join('').trim();

			if (!svgText) return;

			const svgPre: Element = {
				type: 'element',
				tagName: 'pre',
				properties: {
					className: ['svg-diagram']
				},
				children: [{ type: 'text', value: svgText } as Text]
			};

			(parent.children as ElementContent[])[index] = svgPre;
		});
	};
};
