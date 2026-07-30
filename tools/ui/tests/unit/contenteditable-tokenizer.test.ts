import { describe, expect, it } from 'vitest';
import { tokenizeContent } from '$lib/utils';

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
});
