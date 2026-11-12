import { hasMarkdownStructure } from '$lib/components/app/content/MarkdownEditor/markdown-structure';
import { describe, expect, it } from 'vitest';

describe('hasMarkdownStructure', () => {
	it('is false for plain paragraphs, inline markup included', () => {
		expect(hasMarkdownStructure('')).toBe(false);
		expect(hasMarkdownStructure('hello world')).toBe(false);
		expect(hasMarkdownStructure('one\n\ntwo')).toBe(false);
		expect(hasMarkdownStructure('some **bold**, `code` and [a](file:///tmp/a.md)')).toBe(false);
		expect(hasMarkdownStructure('/model is broken')).toBe(false);
	});

	it('is true for any other block', () => {
		expect(hasMarkdownStructure('# Title')).toBe(true);
		expect(hasMarkdownStructure('text\n\n- item')).toBe(true);
		expect(hasMarkdownStructure('1. first')).toBe(true);
		expect(hasMarkdownStructure('> quote')).toBe(true);
		expect(hasMarkdownStructure('```js\nconst a = 1;\n```')).toBe(true);
		expect(hasMarkdownStructure('| a | b |\n| - | - |\n| 1 | 2 |')).toBe(true);
		expect(hasMarkdownStructure('---')).toBe(true);
	});
});
