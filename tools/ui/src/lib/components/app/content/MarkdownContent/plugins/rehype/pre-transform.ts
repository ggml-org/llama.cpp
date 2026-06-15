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
 * Accepts one or more source languages, and an optional contentGuard that
 * receives the trimmed text and decides whether the block qualifies. The guard
 * lets a shared fence language be claimed only when its content matches, e.g.
 * an xml block is converted to svg only when it starts with <svg.
 *
 * The result has no <code> child, so rehypeEnhanceCodeBlocks skips it. Rendering
 * happens client-side, so no markup is injected at this stage. Must run BEFORE
 * rehypeEnhanceCodeBlocks.
 */
export function createPreTransform(
	languages: string | string[],
	targetClass: string,
	contentGuard?: (text: string) => boolean
): Plugin<[], Root> {
	const codeClasses = (Array.isArray(languages) ? languages : [languages]).map(
		(language) => `language-${language}`
	);

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

				const matches = className.some(
					(cls) => typeof cls === 'string' && codeClasses.includes(cls)
				);

				if (!matches) return;

				// Recursively extract text to handle nested spans from syntax highlighting
				const text = codeElement.children.map(extractText).join('').trim();

				if (!text) return;

				if (contentGuard && !contentGuard(text)) return;

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
