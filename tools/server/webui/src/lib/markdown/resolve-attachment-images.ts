import type { Root as HastRoot } from 'hast';
import { visit } from 'unist-util-visit';
import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types/database';
import { AttachmentType, UrlPrefix } from '$lib/enums';

/**
 * Rehype plugin to resolve attachment image sources.
 * Converts attachment names to base64 data URLs.
 */
export function rehypeResolveAttachmentImages(options: { attachments?: DatabaseMessageExtra[] }) {
	return (tree: HastRoot) => {
		visit(tree, 'element', (node) => {
			if (node.tagName === 'img' && node.properties?.src) {
				const src = String(node.properties.src);

				if (src.startsWith(UrlPrefix.DATA) || src.startsWith(UrlPrefix.HTTP)) {
					return;
				}

				const attachment = options.attachments?.find(
					(a): a is DatabaseMessageExtraImageFile =>
						a.type === AttachmentType.IMAGE && a.name === src
				);

				if (attachment?.base64Url) {
					node.properties.src = attachment.base64Url;
				}
			}
		});
	};
}
