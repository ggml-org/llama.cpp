import {
	buildSkillResourceTree,
	classifySkillResourceFormat,
	createSkillRootNode,
	flattenSkillResourceTree,
	getInitialExpandedFolderPaths
} from '$lib/components/app/skills/skill-resource-presentation';
import { normalizeSkillDescription } from '$lib/utils/formatters';
import { describe, expect, it } from 'vitest';

describe('skill resource presentation', () => {
	it.each([
		['SKILL.md', 'markdown'],
		['references/guide.md', 'markdown'],
		['references/guide.markdown', 'markdown'],
		['assets/example.html', 'html'],
		['assets/example.htm', 'html'],
		['scripts/build.ts', 'source'],
		['notes/data.json', 'source'],
		['notes/readme.txt', 'source']
	] as const)('classifies %s as %s', (path, format) => {
		expect(classifySkillResourceFormat(path)).toBe(format);
	});

	it.each([
		'README',
		'unknown.blob',
		'document.pdf',
		'image.png',
		'image.svg',
		'audio.mp3',
		'video.mp4',
		'archive.zip',
		'archive.tar.gz',
		'program.exe'
	])('keeps unsupported resource %s unavailable', (path) => {
		expect(classifySkillResourceFormat(path)).toBe('unsupported');
	});

	it('builds a deduplicated hierarchy without changing relative paths', () => {
		const tree = buildSkillResourceTree([
			'references/API.md',
			'scripts/nested/run.sh',
			'references/API.md',
			'asset.bin'
		]);

		expect(createSkillRootNode()).toEqual({
			format: 'markdown',
			kind: 'file',
			name: 'SKILL.md',
			path: 'SKILL.md'
		});
		expect(tree).toEqual([
			{
				children: [
					{
						format: 'markdown',
						kind: 'file',
						name: 'API.md',
						path: 'references/API.md'
					}
				],
				kind: 'folder',
				name: 'references',
				path: 'references'
			},
			{
				children: [
					{
						children: [
							{
								format: 'source',
								kind: 'file',
								name: 'run.sh',
								path: 'scripts/nested/run.sh'
							}
						],
						kind: 'folder',
						name: 'nested',
						path: 'scripts/nested'
					}
				],
				kind: 'folder',
				name: 'scripts',
				path: 'scripts'
			},
			{
				format: 'unsupported',
				kind: 'file',
				name: 'asset.bin',
				path: 'asset.bin'
			}
		]);
	});

	it('expands top-level folders and flattens only visible descendants', () => {
		const root = createSkillRootNode();
		const tree = buildSkillResourceTree([
			'references/API.md',
			'scripts/nested/run.sh',
			'scripts/top.ts'
		]);
		const expanded = getInitialExpandedFolderPaths(tree);
		const rows = flattenSkillResourceTree([root, ...tree], expanded);

		expect([...expanded]).toEqual(['references', 'scripts']);
		expect(rows.map(({ depth, node }) => [node.path, depth])).toEqual([
			['SKILL.md', 0],
			['references', 0],
			['references/API.md', 1],
			['scripts', 0],
			['scripts/nested', 1],
			['scripts/top.ts', 1]
		]);
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
