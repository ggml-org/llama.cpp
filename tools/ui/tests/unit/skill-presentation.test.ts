import { BookOpenText, FileText, PackageOpen, Terminal } from '@lucide/svelte';
import {
	classifySkillResourcePath,
	groupSkillResourcePaths
} from '$lib/components/app/skills/skill-resource-presentation';
import { normalizeSkillDescription } from '$lib/utils/formatters';
import { describe, expect, it } from 'vitest';

describe('classifySkillResourcePath / groupSkillResourcePaths', () => {
	it('classifies only the first path segment', () => {
		expect(classifySkillResourcePath('scripts/build.py')).toMatchObject({
			group: 'scripts',
			icon: Terminal,
			label: 'Scripts'
		});
		expect(classifySkillResourcePath('references/API.md')).toMatchObject({
			group: 'references',
			icon: BookOpenText,
			label: 'References'
		});
		expect(classifySkillResourcePath('assets/template.docx')).toMatchObject({
			group: 'assets',
			icon: PackageOpen,
			label: 'Assets'
		});
		expect(classifySkillResourcePath('notes/references.md')).toMatchObject({
			group: 'other',
			icon: FileText,
			label: 'Other files'
		});
	});

	it('omits empty groups and keeps the required order', () => {
		expect(
			groupSkillResourcePaths(['misc/data.json', 'assets/sample.svg', 'scripts/run.sh'])
		).toEqual([
			expect.objectContaining({ group: 'scripts', paths: ['scripts/run.sh'] }),
			expect.objectContaining({ group: 'assets', paths: ['assets/sample.svg'] }),
			expect.objectContaining({ group: 'other', paths: ['misc/data.json'] })
		]);
		expect(groupSkillResourcePaths([])).toEqual([]);
	});
});

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