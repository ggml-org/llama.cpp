import { markdownLanguage } from '@codemirror/lang-markdown';
import { LanguageSupport } from '@codemirror/language';
import { EditorState } from '@codemirror/state';
import {
	codeRanges,
	isInCodeBlock
} from '$lib/components/app/content/MarkdownEditor/markdown-syntax';
import { describe, expect, it } from 'vitest';

function stateOf(doc: string): EditorState {
	return EditorState.create({ doc, extensions: new LanguageSupport(markdownLanguage) });
}

function inCode(doc: string, pos = doc.length): boolean {
	return isInCodeBlock(stateOf(doc), pos);
}

describe('isInCodeBlock', () => {
	it('is false in plain text and inline code', () => {
		expect(inCode('hello world', 5)).toBe(false);
		expect(inCode('run `npm test` now', 10)).toBe(false);
	});

	it('is true right after an opening fence and in a still-open block', () => {
		expect(inCode('```')).toBe(true);
		expect(inCode('```js')).toBe(true);
		expect(inCode('```js\nconst a = 1;')).toBe(true);
	});

	it('knows tilde fences and indented code', () => {
		expect(inCode('~~~\ncode')).toBe(true);
		expect(inCode('text\n\n    indented')).toBe(true);
	});

	it('is true inside a closed block and false after or before it', () => {
		const block = '```\ncode\n```';

		expect(inCode(block, 6)).toBe(true);
		expect(inCode(block, 0)).toBe(false);
		expect(inCode(block)).toBe(false);
		expect(inCode(`${block}\nafter`)).toBe(false);
	});

	it('tells two blocks apart', () => {
		const two = '```\na\n```\nmid\n```\nb\n```';
		const second = two.lastIndexOf('```\nb');

		expect(inCode(two, two.indexOf('mid') + 1)).toBe(false);
		expect(inCode(two, second + 5)).toBe(true);
		expect(inCode(two)).toBe(false);
	});
});

describe('codeRanges', () => {
	it('lists inline code and code blocks in range', () => {
		const doc = 'a `b` c\n\n~~~\nd\n~~~';
		const state = stateOf(doc);

		expect(codeRanges(state, 0, doc.length)).toEqual([
			[2, 5],
			[9, doc.length]
		]);
	});
});
