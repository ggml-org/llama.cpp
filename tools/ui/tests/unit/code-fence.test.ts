import { isOffsetInCodeBlock } from '$lib/utils';
import { describe, expect, it } from 'vitest';

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
