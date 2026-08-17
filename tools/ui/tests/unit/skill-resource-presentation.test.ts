import {
	buildSkillResourceTree,
	classifySkillResourceFormat,
	createSkillRootNode,
	findSkillResourceParentPath,
	flattenSkillResourceTree,
	getInitialExpandedFolderPaths
} from '$lib/components/app/skills/skill-resource-presentation';
import { describe, expect, it } from 'vitest';

describe('skill resource presentation', () => {
	it('classifies previewable and unsupported resource formats', () => {
		expect(classifySkillResourceFormat('SKILL.md')).toBe('markdown');
		expect(classifySkillResourceFormat('references/guide.markdown')).toBe('markdown');
		expect(classifySkillResourceFormat('preview/index.htm')).toBe('html');
		expect(classifySkillResourceFormat('scripts/check.py')).toBe('source');
		expect(classifySkillResourceFormat('assets/model.bin')).toBe('unsupported');
		expect(classifySkillResourceFormat('assets/archive.zip')).toBe('unsupported');
	});

	it('builds a stable nested tree without duplicate paths', () => {
		const tree = buildSkillResourceTree([
			'references/guide.md',
			'references/nested/example.txt',
			'notes.txt',
			'references/guide.md'
		]);

		expect(createSkillRootNode()).toMatchObject({ name: 'SKILL.md', format: 'markdown' });
		expect(tree).toMatchObject([
			{
				kind: 'folder',
				name: 'references',
				children: [
					{ kind: 'file', path: 'references/guide.md' },
					{
						kind: 'folder',
						path: 'references/nested',
						children: [{ kind: 'file', path: 'references/nested/example.txt' }]
					}
				]
			},
			{ kind: 'file', path: 'notes.txt' }
		]);
	});

	it('expands only top-level folders and preserves parents in flattened rows', () => {
		const tree = buildSkillResourceTree(['references/guide.md', 'references/nested/example.txt']);
		const expanded = getInitialExpandedFolderPaths(tree);
		const rows = flattenSkillResourceTree(tree, expanded);

		expect(expanded).toEqual(new Set(['references']));
		expect(rows.map(({ node }) => node.path)).toEqual([
			'references',
			'references/guide.md',
			'references/nested'
		]);
		expect(findSkillResourceParentPath(rows, 'references/guide.md')).toBe('references');
	});
});
