import { BookOpenText, FileText, PackageOpen, Terminal } from '@lucide/svelte';
import type { Component } from 'svelte';

export type SkillResourceGroup = 'scripts' | 'references' | 'assets' | 'other';

export interface SkillResourcePresentation {
	group: SkillResourceGroup;
	label: string;
	icon: Component;
}

export interface SkillResourceGroupPresentation extends SkillResourcePresentation {
	paths: string[];
}

const RESOURCE_GROUPS: readonly SkillResourcePresentation[] = [
	{ group: 'scripts', icon: Terminal, label: 'Scripts' },
	{ group: 'references', icon: BookOpenText, label: 'References' },
	{ group: 'assets', icon: PackageOpen, label: 'Assets' },
	{ group: 'other', icon: FileText, label: 'Other files' }
];

export function classifySkillResourcePath(path: string): SkillResourcePresentation {
	const firstSegment = path.split('/', 1)[0];

	return RESOURCE_GROUPS.find(({ group }) => group === firstSegment) ?? RESOURCE_GROUPS[3];
}

export function groupSkillResourcePaths(
	paths: readonly string[]
): SkillResourceGroupPresentation[] {
	return RESOURCE_GROUPS.flatMap((presentation) => {
		const groupedPaths = paths.filter(
			(path) => classifySkillResourcePath(path).group === presentation.group
		);

		return groupedPaths.length === 0 ? [] : [{ ...presentation, paths: groupedPaths }];
	});
}
