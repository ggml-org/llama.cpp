/**
 * Rehype plugin that highlights code blocks with rehype-highlight, loading highlight.js only when a message has one.
 *
 * Only `<pre><code>` blocks are highlighted, so a tree without one is left alone and a chat without code never
 * downloads highlight.js.
 */

import type { Element, Root } from 'hast';
import type { Options } from 'rehype-highlight';
import type { Plugin, Transformer } from 'unified';
import { visit } from 'unist-util-visit';

let highlightModules: Promise<typeof import('./lazy-highlight-modules')> | undefined;

function hasCodeBlock(tree: Root): boolean {
	let found = false;

	visit(tree, 'element', (node: Element) => {
		if (
			node.tagName === 'pre' &&
			node.children.some((c) => c.type === 'element' && c.tagName === 'code')
		) {
			found = true;

			return false;
		}
	});

	return found;
}

// synchronous when there is no code block or highlight.js has loaded, so only the first message with code waits for
// it, and every other one renders in the same pass as before
export const rehypeLazyHighlight: Plugin<[Omit<Options, 'languages'>?], Root> = (options = {}) => {
	let transformer: Transformer<Root> | undefined;

	return (tree, file) => {
		if (!hasCodeBlock(tree)) return;

		// rehype-highlight changes the tree in place
		if (transformer) {
			transformer(tree, file, () => {});

			return;
		}

		highlightModules ??= import('./lazy-highlight-modules').catch((error) => {
			// let a later message try again
			highlightModules = undefined;

			throw error;
		});

		return highlightModules.then(({ languages, rehypeHighlight }) => {
			transformer ??= rehypeHighlight({ ...options, languages }) as Transformer<Root>;
			transformer(tree, file, () => {});
		});
	};
};
