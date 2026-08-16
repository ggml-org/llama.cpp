import { BookOpenText, FileText, PackageOpen, Terminal } from '@lucide/svelte';
import {
	classifySkillResourcePath,
	groupSkillResourcePaths
} from '$lib/components/app/skills/skill-resource-presentation';
import { expect, it } from 'vitest';

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
