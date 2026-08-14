import { BookOpenText, FileText, PackageOpen, Terminal } from '@lucide/svelte';
import { expect, it } from 'vitest';
import {
	classifySkillResourcePath,
	groupSkillResourcePaths
} from '$lib/components/app/skills/skill-resource-presentation';

it('classifies only the first path segment', () => {
	expect(classifySkillResourcePath('scripts/build.py')).toMatchObject({
		group: 'scripts', label: 'Scripts', icon: Terminal
	});
	expect(classifySkillResourcePath('references/API.md')).toMatchObject({
		group: 'references', label: 'References', icon: BookOpenText
	});
	expect(classifySkillResourcePath('assets/template.docx')).toMatchObject({
		group: 'assets', label: 'Assets', icon: PackageOpen
	});
	expect(classifySkillResourcePath('notes/references.md')).toMatchObject({
		group: 'other', label: 'Other files', icon: FileText
	});
});

it('omits empty groups and keeps the required order', () => {
	expect(groupSkillResourcePaths(['misc/data.json', 'assets/sample.svg', 'scripts/run.sh']))
		.toEqual([
			expect.objectContaining({ group: 'scripts', paths: ['scripts/run.sh'] }),
			expect.objectContaining({ group: 'assets', paths: ['assets/sample.svg'] }),
			expect.objectContaining({ group: 'other', paths: ['misc/data.json'] })
		]);
	expect(groupSkillResourcePaths([])).toEqual([]);
});
