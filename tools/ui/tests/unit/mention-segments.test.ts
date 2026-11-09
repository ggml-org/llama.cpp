import { splitMentionSegments } from '$lib/utils/mention-badge';
import { describe, expect, it } from 'vitest';

describe('splitMentionSegments', () => {
	it('returns a single plain run when there is no mention', () => {
		const segments = splitMentionSegments('# not a heading here');

		expect(segments).toEqual([{ mention: null, text: '# not a heading here' }]);
	});

	it('splits text around a mention', () => {
		const segments = splitMentionSegments('look at [main.c](file:///src/main.c) please');

		expect(segments.map((segment) => segment.text)).toEqual([
			'look at ',
			'[main.c](file:///src/main.c)',
			' please'
		]);
		expect(segments[1].mention).toEqual({ name: 'main.c', path: '/src/main.c' });
	});

	it('decodes percent-encoded paths', () => {
		const segments = splitMentionSegments('[a b.txt](file:///tmp/a%20b.txt)');

		expect(segments[0].mention?.path).toBe('/tmp/a b.txt');
	});

	it('keeps the directory marker so the folder icon is picked', () => {
		const segments = splitMentionSegments('[src](file:///repo/src/)');

		expect(segments[0].mention?.path).toBe('/repo/src/');
	});

	it('handles adjacent mentions with no text between them', () => {
		const segments = splitMentionSegments('[a](file:///a)[b](file:///b)');

		expect(segments).toHaveLength(2);
		expect(segments.every((segment) => segment.mention !== null)).toBe(true);
	});

	it('preserves the exact source when segments are joined back', () => {
		const source = 'see [a](file:///a) and [b](file:///b/) done';

		expect(splitMentionSegments(source).reduce((acc, segment) => acc + segment.text, '')).toBe(
			source
		);
	});
});
