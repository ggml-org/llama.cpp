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
 * Builds a rehype plugin that converts <pre><code class="language-{language}">
 * blocks into <pre class="{targetClass}"> elements carrying the raw text.
 *
 * Transforms:
 *   <pre><code class="language-mermaid">graph TD; A-->B</code></pre>
 * into:
 *   <pre class="mermaid">graph TD; A-->B</pre>
 *
 * The result has no <code> child, so rehypeEnhanceCodeBlocks skips it. Rendering
 * happens client-side, so no markup is injected at this stage. Must run BEFORE
 * rehypeEnhanceCodeBlocks.
 */
export function createPreTransform(language: string, targetClass: string): Plugin<[], Root> {
	const codeClass = `language-${language}`;

	return () => {
		return (tree: Root) => {
			visit(tree, 'element', (node: Element, index, parent) => {
				if (node.tagName !== 'pre' || !parent || index === undefined) return;

				const codeElement = node.children.find(
					(child): child is Element => child.type === 'element' && child.tagName === 'code'
				);

				if (!codeElement) return;

				const className = codeElement.properties?.className;
				if (!Array.isArray(className)) return;

				const matches = className.some((cls) => typeof cls === 'string' && cls === codeClass);

				if (!matches) return;

				// Recursively extract text to handle nested spans from syntax highlighting
				const text = codeElement.children.map(extractText).join('').trim();

				if (!text) return;

				const pre: Element = {
					type: 'element',
					tagName: 'pre',
					properties: {
						className: [targetClass]
					},
					children: [{ type: 'text', value: text } as Text]
				};

				(parent.children as ElementContent[])[index] = pre;
			});
		};
	};
}
