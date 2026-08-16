import { normalizeSkillDescription } from '$lib/utils/formatters';
import { describe, expect, it } from 'vitest';

describe('normalizeSkillDescription', () => {
	it('collapses repeated spaces and tabs into single spaces', () => {
		expect(normalizeSkillDescription('hello   world')).toBe('hello world');
		expect(normalizeSkillDescription('hello\t\tworld')).toBe('hello world');
		expect(normalizeSkillDescription('a \t b \t\t c')).toBe('a b c');
	});

	it('trims leading and trailing whitespace', () => {
		expect(normalizeSkillDescription('  leading and trailing  ')).toBe('leading and trailing');
		expect(normalizeSkillDescription('\t\npadded\t\n')).toBe('padded');
	});

	it('collapses literal multiline text with indentation and blank lines', () => {
		const description = `Usage:
	Run the tool with a query.

		The query may span lines.

	Results are returned inline.`;

		expect(normalizeSkillDescription(description)).toBe(
			'Usage: Run the tool with a query. The query may span lines. Results are returned inline.'
		);
	});

	it('collapses folded-style YAML line breaks', () => {
		expect(
			normalizeSkillDescription('This long paragraph is split\nacross multiple lines in YAML.')
		).toBe('This long paragraph is split across multiple lines in YAML.');
	});

	it('returns an empty string for whitespace-only input', () => {
		expect(normalizeSkillDescription('')).toBe('');
		expect(normalizeSkillDescription('   ')).toBe('');
		expect(normalizeSkillDescription('\t\n \t ')).toBe('');
	});
});
