import { describe, expect, it } from 'vitest';
import {
	extractSearchResults,
	extractSearchQuery,
	faviconForUrl,
	isWebSearchToolName
} from '$lib/utils/search-results';

describe('extractSearchResults', () => {
	it('parses the Exa fixture with multiple results', () => {
		const fixture = `Title: World Cup 2026 | Match schedule, fixtures
URL: https://www.fifa.com/articles/match-schedule
Published: 2026-06-22T00:01:00.000Z
Author: N/A
Highlights:
Find out the full match schedule for World Cup 2026
---
Title: 2026 FIFA World Cup match schedule
URL: https://www.espn.com/soccer/story/abc/def
Published: 2026-07-08T07:07:00.000Z
Author: ESPN
Highlights:
Round of 32 · Tuesday, July 7
---
Title: BBC
URL: https://www.bbc.co.uk/sport/football/world-cup/schedule
Published: N/A
Author: N/A
Highlights:
# FIFA World Cup Schedule
...`;
		const results = extractSearchResults(fixture);
		expect(results.length).toBe(3);
		expect(results[0].title).toContain('World Cup 2026');
		expect(results[0].url).toBe('https://www.fifa.com/articles/match-schedule');
		expect(results[0].published).toBe('2026-06-22T00:01:00.000Z');
		// N/A filtered out
		expect(results[0].author).toBeUndefined();
		expect(results[0].highlights).toContain('match schedule');
		expect(results[1].author).toBe('ESPN');
		expect(results[2].author).toBeUndefined(); // N/A filtered
	});

	it('returns empty array for empty input', () => {
		expect(extractSearchResults('')).toEqual([]);
		expect(extractSearchResults(undefined)).toEqual([]);
		expect(extractSearchResults(null)).toEqual([]);
	});

	it('skips chunks missing title or url', () => {
		const txt = `Title: no url here
Highlights:
foo
---
Title: foo
URL: https://x.com
---
just a paragraph
---
Title: b
URL: not a url`;
		const results = extractSearchResults(txt);
		// Only middle one should pass (has title + url).
		expect(results.length).toBe(1);
		expect(results[0].url).toBe('https://x.com');
	});

	it('parses a single result without separators', () => {
		const txt = `Title: only one
URL: https://example.com/test
Published: 2026-01-01T00:00:00Z
Author: alice
Highlights:
a highlight`;
		const results = extractSearchResults(txt);
		expect(results.length).toBe(1);
		expect(results[0].title).toBe('only one');
		expect(results[0].author).toBe('alice');
		expect(results[0].highlights).toBe('a highlight');
	});

	it('extracts query from JSON toolArgs', () => {
		expect(extractSearchQuery('{"query":"foo"}')).toBe('foo');
		expect(extractSearchQuery('  {"query":"  foo  "}  ')).toBe('foo');
		expect(extractSearchQuery('not json')).toBe('');
		expect(extractSearchQuery(null)).toBe('');
		expect(extractSearchQuery('{"query":123}')).toBe('');
	});

	it('resolves favicon URLs from origins', () => {
		expect(faviconForUrl('https://example.com/path/to/page')).toBe(
			'https://example.com/favicon.ico'
		);
		expect(faviconForUrl('http://example.com/x')).toBe('http://example.com/favicon.ico');
		expect(faviconForUrl('not a url')).toBeNull();
	});
});

describe('isWebSearchToolName', () => {
	it('excludes tools that take the same query argument but are not web searches', () => {
		expect(isWebSearchToolName('search_pull_requests')).toBe(false);
		expect(isWebSearchToolName('search_code')).toBe(false);
		expect(isWebSearchToolName('search_repositories')).toBe(false);
		expect(isWebSearchToolName('search_issues')).toBe(false);
	});

	it('handles empty / missing input', () => {
		expect(isWebSearchToolName(null)).toBe(false);
		expect(isWebSearchToolName(undefined)).toBe(false);
		expect(isWebSearchToolName('')).toBe(false);
	});

	it('returns false for unrelated tools', () => {
		expect(isWebSearchToolName('web_fetch')).toBe(false);
		expect(isWebSearchToolName('read_file')).toBe(false);
		expect(isWebSearchToolName('exec_shell_command')).toBe(false);
	});
});
