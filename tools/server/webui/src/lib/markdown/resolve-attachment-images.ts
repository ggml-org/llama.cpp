import type { Root as HastRoot } from 'hast';
import { visit } from 'unist-util-visit';
import type { DatabaseMessage, DatabaseMessageExtraImageFile } from '$lib/types/database';

/**
 * Rehype plugin to resolve attachment image sources.
 * Converts attachment names (e.g., "mcp-attachment-xxx.png") to base64 data URLs.
 */
export function rehypeResolveAttachmentImages(options: { message?: DatabaseMessage }) {
	return (tree: HastRoot) => {
		visit(tree, 'element', (node) => {
			if (node.tagName === 'img' && node.properties?.src) {
				const src = String(node.properties.src);

				// Skip data URLs and external URLs
				if (src.startsWith('data:') || src.startsWith('http')) {
					return;
				}

				// Find matching attachment
				const attachment = options.message?.extra?.find(
					(a): a is DatabaseMessageExtraImageFile => a.type === 'IMAGE' && a.name === src
				);

				// Replace with base64 URL if found
				if (attachment?.base64Url) {
					node.properties.src = attachment.base64Url;
				}
			}
		});
	};
}
