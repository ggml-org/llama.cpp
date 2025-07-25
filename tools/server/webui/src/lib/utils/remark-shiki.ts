import { visit } from 'unist-util-visit';
import type { Plugin } from 'unified';
import type { Root, Code, Parent } from 'mdast';
import { highlightCode, getThemeForMode } from './syntax-highlighting';

interface RemarkShikiOptions {
	theme?: 'light' | 'dark' | 'auto';
}

// Remark plugin to add Shiki syntax highlighting
export const remarkShiki: Plugin<[RemarkShikiOptions?], Root> = (options = {}) => {
	return async (tree: Root) => {
		const codeNodes: Array<{ node: Code; index: number; parent: Parent }> = [];
		
		// Collect all code nodes
		visit(tree, 'code', (node: Code, index, parent) => {
			if (index !== undefined && parent) {
				codeNodes.push({ node, index, parent });
			}
		});

		// Process each code node with syntax highlighting
		for (const { node, index, parent } of codeNodes) {
			try {
				// Determine theme based on options
				const isDark = options.theme === 'dark' || 
					(options.theme === 'auto' && 
					 typeof window !== 'undefined' && 
					 window.matchMedia('(prefers-color-scheme: dark)').matches);
				
				const theme = getThemeForMode(isDark);
				
				// Normalize the code content at the remark level
				const normalizedCode = node.value.replace(/^\n+/, '').replace(/\n+$/, '');
				
				// Highlight the code
				const highlightedHtml = await highlightCode(
					normalizedCode,
					node.lang || undefined,
					theme
				);

				// Replace the code node with an HTML node
				const htmlNode = {
					type: 'html' as const,
					value: highlightedHtml
				};

				parent.children[index] = htmlNode as any;
			} catch (error) {
				console.error('Failed to highlight code block:', error);
				// Keep the original code node if highlighting fails
			}
		}
	};
};

export default remarkShiki;
