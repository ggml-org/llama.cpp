/**
 * Rehype plugin that renders math with rehype-katex, loading KaTeX only when a message has math.
 *
 * remark-math marks math as elements with a `language-math`, `math-inline` or `math-display`
 * class; a tree without one is left alone, so a chat without math never downloads KaTeX.
 */

import type { Element, Root } from 'hast';
import type { Plugin, Transformer } from 'unified';
import { visit } from 'unist-util-visit';

const MATH_CLASSES = ['language-math', 'math-inline', 'math-display'];

let katexLoading: Promise<Transformer<Root>> | undefined;
let katexTransformer: Transformer<Root> | undefined;

function hasMath(tree: Root): boolean {
	let found = false;

	visit(tree, 'element', (node: Element) => {
		const className = node.properties?.className;

		if (Array.isArray(className) && className.some((c) => MATH_CLASSES.includes(String(c)))) {
			found = true;

			return false;
		}
	});

	return found;
}

// synchronous when there is no math or KaTeX has loaded, so only the first message with math
// waits for it, and every other one renders in the same pass as before
export const rehypeLazyKatex: Plugin<[], Root> = () => {
	return (tree, file) => {
		if (!hasMath(tree)) return;

		// rehype-katex changes the tree in place
		if (katexTransformer) {
			katexTransformer(tree, file, () => {});

			return;
		}

		katexLoading ??= import('rehype-katex').then(({ default: rehypeKatex }) => {
			katexTransformer = rehypeKatex() as Transformer<Root>;

			return katexTransformer;
		});

		return katexLoading.then((transform) => {
			transform(tree, file, () => {});
		});
	};
};
