import { describe, expect, it } from 'vitest';
import { containsCodeSpan, isOffsetInCodeBlock, tokenizeContent } from '$lib/utils';

describe('tokenizeContent', () => {
	it('tokenizes a plain text buffer with no badges', () => {
		expect(tokenizeContent('hello world')).toEqual([{ kind: 'text', text: 'hello world' }]);
	});

	it('tokenizes a single badge', () => {
		expect(tokenizeContent('[docs](file:///a/b)')).toEqual([
			{ kind: 'badge', name: 'docs', path: '/a/b' }
		]);
	});

	it('tokenizes text around a single badge', () => {
		expect(tokenizeContent('hello [docs](file:///a/b) world')).toEqual([
			{ kind: 'text', text: 'hello ' },
			{ kind: 'badge', name: 'docs', path: '/a/b' },
			{ kind: 'text', text: ' world' }
		]);
	});

	it('tokenizes adjacent badges as separate tokens', () => {
		expect(tokenizeContent('[a](file:///x)[b](file:///y)')).toEqual([
			{ kind: 'badge', name: 'a', path: '/x' },
			{ kind: 'badge', name: 'b', path: '/y' }
		]);
	});

	it('leaves non-file links untouched in the stream', () => {
		expect(tokenizeContent('see [foo](https://example.com) for details')).toEqual([
			{ kind: 'text', text: 'see [foo](https://example.com) for details' }
		]);
	});

	it('recognizes badges whose path contains spaces (macOS screenshots)', () => {
		const path = '/Users/allozaur/Desktop/Screenshot 2026-07-28 at 17.21.50.png';
		const source = `[Screenshot 2026-07-28 at 17.21.50.png](file://${path}) `;
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'badge', name: 'Screenshot 2026-07-28 at 17.21.50.png', path },
			{ kind: 'text', text: ' ' }
		]);
	});

	it('recognizes badges whose path lives in the macOS temp folder', () => {
		const path =
			'/var/folders/78/j28m7pn57wb34bfjwlskh62h0000gn/T/TemporaryItems/NSIRD_screencaptureui_GD0A2R/Screenshot 2026-07-28 at 17.23.28.png';
		const source = `[Screenshot 2026-07-28 at 17.23.28.png](file://${path}) `;
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'badge', name: 'Screenshot 2026-07-28 at 17.23.28.png', path },
			{ kind: 'text', text: ' ' }
		]);
	});

	it('keeps text around a badge with spaces in the path', () => {
		const path = '/Users/allozaur/Desktop/Screenshot 2026-07-28 at 17.21.50.png';
		const source = `see [Screenshot 2026-07-28 at 17.21.50.png](file://${path}) done`;
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'text', text: 'see ' },
			{ kind: 'badge', name: 'Screenshot 2026-07-28 at 17.21.50.png', path },
			{ kind: 'text', text: ' done' }
		]);
	});

	it('recognizes badges whose path contains a close parenthesis (macOS duplicate files)', () => {
		const path = '/Users/foo/Screenshot (1).png';
		const source = `[Screenshot (1).png](file://${path}) `;
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'badge', name: 'Screenshot (1).png', path },
			{ kind: 'text', text: ' ' }
		]);
	});

	it('recognizes badges whose folder name is wrapped in parentheses', () => {
		const path = '/Users/foo/Project (Stuff)/main.rs';
		const source = `[main.rs](file://${path}) `;
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'badge', name: 'main.rs', path },
			{ kind: 'text', text: ' ' }
		]);
	});

	it('recognizes adjacent badges back-to-back with no separator', () => {
		const source = '[a](file:///p)[b](file:///q)';
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'badge', name: 'a', path: '/p' },
			{ kind: 'badge', name: 'b', path: '/q' }
		]);
	});

	it('tokenizes inline code with the backticks included', () => {
		expect(tokenizeContent('run `npm test` now')).toEqual([
			{ kind: 'text', text: 'run ' },
			{ kind: 'inlineCode', text: '`npm test`' },
			{ kind: 'text', text: ' now' }
		]);
	});

	it('tokenizes a fenced code block without a language', () => {
		const source = 'before\n```\nconst a = 1;\n```\nafter';
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'text', text: 'before\n' },
			{ kind: 'codeBlock', text: '```\nconst a = 1;\n```' },
			{ kind: 'text', text: '\nafter' }
		]);
	});

	it('tokenizes a fenced code block with a language', () => {
		const source = '```js\nconst a = 1;\n```';
		expect(tokenizeContent(source)).toEqual([
			{ kind: 'codeBlock', text: '```js\nconst a = 1;\n```' }
		]);
	});

	it('prefers the fenced block over inline spans at triple backticks', () => {
		expect(tokenizeContent('```a``` ```b```')).toEqual([
			{ kind: 'codeBlock', text: '```a```' },
			{ kind: 'text', text: ' ' },
			{ kind: 'codeBlock', text: '```b```' }
		]);
	});

	it('leaves an unclosed fence as plain text', () => {
		expect(tokenizeContent('```js\nconst a = 1;')).toEqual([
			{ kind: 'text', text: '```js\nconst a = 1;' }
		]);
	});

	it('leaves an unclosed inline backtick as plain text', () => {
		expect(tokenizeContent('run `npm test')).toEqual([{ kind: 'text', text: 'run `npm test' }]);
	});

	it('does not recognize badges inside code spans', () => {
		expect(tokenizeContent('`[a](file:///p)`')).toEqual([
			{ kind: 'inlineCode', text: '`[a](file:///p)`' }
		]);
	});

	it('tokenizes badges and code spans side by side', () => {
		expect(tokenizeContent('[a](file:///p) `x`')).toEqual([
			{ kind: 'badge', name: 'a', path: '/p' },
			{ kind: 'text', text: ' ' },
			{ kind: 'inlineCode', text: '`x`' }
		]);
	});
});

describe('containsCodeSpan', () => {
	it('detects inline code', () => {
		expect(containsCodeSpan('run `npm test` now')).toBe(true);
	});

	it('detects a fenced block with a language', () => {
		expect(containsCodeSpan('```js\nconst a = 1;\n```')).toBe(true);
	});

	it('detects a fenced block without a language', () => {
		expect(containsCodeSpan('```\ncode\n```')).toBe(true);
	});

	it('ignores unclosed fences and lone backticks', () => {
		expect(containsCodeSpan('```js\nconst a = 1;')).toBe(false);
		expect(containsCodeSpan('run `npm test')).toBe(false);
		expect(containsCodeSpan('``')).toBe(false);
	});

	it('ignores plain text and mention links', () => {
		expect(containsCodeSpan('hello world')).toBe(false);
		expect(containsCodeSpan('[a](file:///p)')).toBe(false);
	});
});

describe('isOffsetInCodeBlock', () => {
	const BLOCK = '```js\nconst a = 1;\n```';

	it('is false with no fences in the buffer', () => {
		expect(isOffsetInCodeBlock('hello world', 5)).toBe(false);
		expect(isOffsetInCodeBlock('run `npm test` now', 10)).toBe(false);
	});

	it('is true right after the opening fence, before any content', () => {
		expect(isOffsetInCodeBlock('```', 3)).toBe(true);
		expect(isOffsetInCodeBlock('```js', 5)).toBe(true);
	});

	it('is true inside a still-open block while it is being typed', () => {
		const open = '```js\nconst a = 1;';
		expect(isOffsetInCodeBlock(open, open.length)).toBe(true);
	});

	it('is true inside a closed block and false outside it', () => {
		expect(isOffsetInCodeBlock(BLOCK, 6)).toBe(true);
		expect(isOffsetInCodeBlock(BLOCK, 0)).toBe(false);
		expect(isOffsetInCodeBlock(BLOCK, BLOCK.length)).toBe(false);
		expect(isOffsetInCodeBlock(BLOCK + '\nafter', BLOCK.length + 5)).toBe(false);
	});

	it('toggles per fence across multiple blocks', () => {
		const two = BLOCK + '\ntext\n' + BLOCK;
		const secondBlock = two.lastIndexOf(BLOCK);
		expect(isOffsetInCodeBlock(two, secondBlock - 2)).toBe(false);
		expect(isOffsetInCodeBlock(two, secondBlock + 6)).toBe(true);
		expect(isOffsetInCodeBlock(two, two.length)).toBe(false);
	});
});
