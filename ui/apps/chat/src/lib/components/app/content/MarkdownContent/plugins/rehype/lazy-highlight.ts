/**
 * Rehype plugin that highlights code blocks like rehype-highlight, loading highlight.js and each language only when a
 * message has a block in it.
 *
 * Only `<pre><code>` blocks with a language class (`language-ts`, `lang-ts`) are highlighted; `no-highlight` opts a
 * block out, and a language highlight.js lacks is left plain. rehype-highlight itself is not used: it bundles
 * lowlight's common languages, which would load all at once.
 */

import {
	getLoadedHighlightLanguages,
	loadHighlightLanguages,
	resolveHighlightLanguage
} from '#lib/utils/code.ts';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

type Lowlight = ReturnType<(typeof import('./lowlight-core'))['createLowlight']>;

export interface LazyHighlightOptions {
	/** Extra aliases per language, as rehype-highlight takes them: `{ xml: ['svelte', 'vue'] }`. */
	aliases?: Record<string, string | readonly string[]>;
}

interface CodeBlock {
	node: Element;
	/** The highlight.js language to use, or undefined for one it lacks. */
	name: string | undefined;
}

let lowlight: Lowlight | undefined;
let lowlightLoading: Promise<Lowlight> | undefined;

const registered = new Set<string>();

// the language a code element's class names, false for `no-highlight`, as rehype-highlight reads it
function classLanguage(node: Element): string | false | undefined {
	const list = node.properties?.className;

	if (!Array.isArray(list)) return undefined;

	let name: string | undefined;

	for (const item of list) {
		const value = String(item);

		if (value === 'no-highlight' || value === 'nohighlight') return false;

		if (!name && value.startsWith('lang-')) name = value.slice(5);

		if (!name && value.startsWith('language-')) name = value.slice(9);
	}

	return name;
}

function textContent(node: ElementContent | Element): string {
	if (node.type === 'text') return node.value;

	if (node.type !== 'element') return '';

	return node.children.map(textContent).join('');
}

function loadLowlight(): Promise<Lowlight> {
	lowlightLoading ??= import('./lowlight-core').then(
		({ createLowlight }) => (lowlight = createLowlight()),
		(error) => {
			lowlightLoading = undefined;

			throw error;
		}
	);

	return lowlightLoading;
}

// registers the languages highlightCode's loader has loaded with lowlight, as highlight.js ships them
function registerLoadedLanguages(target: Lowlight): void {
	for (const [name, definition] of Object.entries(getLoadedHighlightLanguages())) {
		if (registered.has(name)) continue;

		target.register(name, definition);
		registered.add(name);
	}
}

function highlightBlocks(target: Lowlight, blocks: CodeBlock[]): void {
	for (const { name, node } of blocks) {
		const className = Array.isArray(node.properties.className)
			? node.properties.className
			: (node.properties.className = []);

		if (!className.includes('hljs')) className.unshift('hljs');

		if (!name || !registered.has(name)) continue;

		const result = target.highlight(name, textContent(node));

		if (result.children.length > 0) node.children = result.children as ElementContent[];
	}
}

// synchronous when there is no code block or its languages have loaded, so a message waits only for a language it
// is the first to use, and every other one renders in the same pass as before
export const rehypeLazyHighlight: Plugin<[LazyHighlightOptions?], Root> = (options = {}) => {
	const extraAliases = new Map<string, string>();

	for (const [language, aliases] of Object.entries(options.aliases ?? {})) {
		for (const alias of [aliases].flat()) extraAliases.set(alias.toLowerCase(), language);
	}

	return (tree) => {
		const blocks: CodeBlock[] = [];

		visit(tree, 'element', (node: Element, _, parent) => {
			if (node.tagName !== 'code' || parent?.type !== 'element' || parent.tagName !== 'pre') {
				return;
			}

			const language = classLanguage(node);

			if (!language) return;

			const key = language.toLowerCase();

			blocks.push({ name: resolveHighlightLanguage(extraAliases.get(key) ?? key), node });
		});

		if (!blocks.length) return;

		const names = [
			...new Set(blocks.map((block) => block.name).filter((name) => name !== undefined))
		];

		if (lowlight && names.every((name) => registered.has(name))) {
			highlightBlocks(lowlight, blocks);

			return;
		}

		return Promise.all([loadLowlight(), loadHighlightLanguages(names)]).then(([target]) => {
			registerLoadedLanguages(target);
			highlightBlocks(target, blocks);
		});
	};
};
