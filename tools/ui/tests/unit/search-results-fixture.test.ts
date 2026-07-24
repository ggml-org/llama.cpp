import { describe, it, expect } from 'vitest';
import { extractSearchResults, extractSearchQuery } from '$lib/utils/search-results';

const SAMPLE = `Title: World Cup 2026 | Match schedule, fixtures, results & stadiums
URL: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums
Published: 2026-06-22T00:01:00.000Z
Author: N/A
Highlights:
Find out the full match schedule for World Cup 2026 in Canada, Mexico and USA with fixtures and results from each of the 104 games in the ...
---
Title: 2026 FIFA World Cup match schedule: Fixtures, results, features - ESPN
URL: https://www.espn.com/soccer/story/_/id/48939282/2026-fifa-world-cup-fixtures-results-match-schedule-group-stage-knockout-rounds-bracket
Published: 2026-07-08T07:07:00.000Z
Author: ESPN
Highlights:
Round of 32 · Tuesday, July 7 · Argentina 3-2 Egypt (Atlanta) Switzerland (4) 0-0 (3) Colombia (Vancouver, Canada) · Monday, July 6 · Portugal 0-1 ...
---
Title: BBC
URL: https://www.bbc.co.uk/sport/football/world-cup/schedule
Published: N/A
Author: N/A
Highlights:
Something
# World Cup
...`;

const QUERY_ARGS = '{"query":"FIFA World Cup 2026 schedule"}';

describe('real-world Exa fixture', () => {
	it('extracts every search result and preserves rich highlights', () => {
		const results = extractSearchResults(SAMPLE);
		expect(results.length).toBe(3);
		expect(results[0].title).toContain('World Cup 2026 | Match schedule');
		expect(results[0].url).toContain('fifa.com');
		expect(results[1].title).toContain('2026 FIFA World Cup match schedule');
		expect(results[1].author).toBe('ESPN');
		expect(results[1].highlights).toContain('Round of 32');
		expect(results[2].title).toBe('BBC');
	});

	it('parses the query out of the tool-args JSON', () => {
		expect(extractSearchQuery(QUERY_ARGS)).toBe('FIFA World Cup 2026 schedule');
	});
});
