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
	{ group: 'scripts', label: 'Scripts', icon: Terminal },
	{ group: 'references', label: 'References', icon: BookOpenText },
	{ group: 'assets', label: 'Assets', icon: PackageOpen },
	{ group: 'other', label: 'Other files', icon: FileText }
];

export function classifySkillResourcePath(path: string): SkillResourcePresentation {
	const firstSegment = path.split('/', 1)[0];
	return RESOURCE_GROUPS.find(({ group }) => group === firstSegment) ?? RESOURCE_GROUPS[3];
}

export function groupSkillResourcePaths(paths: readonly string[]): SkillResourceGroupPresentation[] {
	return RESOURCE_GROUPS.flatMap((presentation) => {
		const groupedPaths = paths.filter((path) => classifySkillResourcePath(path).group === presentation.group);
		return groupedPaths.length === 0 ? [] : [{ ...presentation, paths: groupedPaths }];
	});
}
