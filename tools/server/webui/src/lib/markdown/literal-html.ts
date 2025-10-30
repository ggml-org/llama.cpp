import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Break, Content, Paragraph, PhrasingContent, Root, Text } from 'mdast';
import { LINE_BREAK, NBSP, PHRASE_PARENTS, TAB_AS_SPACES } from '$lib/constants/literal-html';

/**
 * remark plugin that rewrites raw HTML nodes into plain-text equivalents.
 *
 * remark parses inline HTML into `html` nodes even when we do not want to render
 * them. We turn each of those nodes into regular text (plus `<br>` break markers)
 * so the downstream rehype pipeline escapes the characters instead of executing
 * them. Leading spaces and tab characters are converted to non‑breaking spaces to
 * keep indentation identical to the original author input.
 */

function preserveIndent(line: string): string {
	let index = 0;
	let output = '';

	while (index < line.length) {
		const char = line[index];

		if (char === ' ') {
			output += NBSP;
			index += 1;
			continue;
		}

		if (char === '\t') {
			output += TAB_AS_SPACES;
			index += 1;
			continue;
		}

		break;
	}

	return output + line.slice(index);
}

function createLiteralChildren(value: string): PhrasingContent[] {
	const lines = value.split(LINE_BREAK);
	const nodes: PhrasingContent[] = [];

	for (const [lineIndex, rawLine] of lines.entries()) {
		if (lineIndex > 0) {
			nodes.push({ type: 'break' } as Break as unknown as PhrasingContent);
		}

		nodes.push({
			type: 'text',
			value: preserveIndent(rawLine)
		} as Text as unknown as PhrasingContent);
	}

	if (!nodes.length) {
		nodes.push({ type: 'text', value: '' } as Text as unknown as PhrasingContent);
	}

	return nodes;
}

export const remarkLiteralHtml: Plugin<[], Root> = () => {
	return (tree) => {
		visit(tree, 'html', (node, index, parent) => {
			if (!parent || typeof index !== 'number') {
				return;
			}

			const replacement = createLiteralChildren(node.value);

			if (!PHRASE_PARENTS.has(parent.type as string)) {
				const paragraph: Paragraph = {
					type: 'paragraph',
					children: replacement as Paragraph['children'],
					data: { literalHtml: true }
				};

				const siblings = parent.children as unknown as Content[];
				siblings.splice(index, 1, paragraph as unknown as Content);

				if (index > 0) {
					const previous = siblings[index - 1] as Paragraph | undefined;

					if (
						previous?.type === 'paragraph' &&
						(previous.data as { literalHtml?: boolean } | undefined)?.literalHtml
					) {
						const prevChildren = previous.children as unknown as PhrasingContent[];

						if (prevChildren.length) {
							const lastChild = prevChildren[prevChildren.length - 1];

							if (lastChild.type !== 'break') {
								prevChildren.push({
									type: 'break'
								} as Break as unknown as PhrasingContent);
							}
						}

						prevChildren.push(...(paragraph.children as unknown as PhrasingContent[]));

						siblings.splice(index, 1);

						return index;
					}
				}

				return index + 1;
			}

			(parent.children as unknown as PhrasingContent[]).splice(
				index,
				1,
				...(replacement as unknown as PhrasingContent[])
			);

			return index + replacement.length;
		});
	};
};
