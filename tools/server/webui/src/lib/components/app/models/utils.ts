import { SvelteMap } from 'svelte/reactivity';
import type { ModelOption } from '$lib/types/models';

export interface ModelItem {
	option: ModelOption;
	flatIndex: number;
}

export interface OrgGroup {
	orgName: string | null;
	items: ModelItem[];
}

export interface GroupedModelOptions {
	loaded: ModelItem[];
	favourites: ModelItem[];
	available: OrgGroup[];
}

export function filterModelOptions(options: ModelOption[], searchTerm: string): ModelOption[] {
	const term = searchTerm.trim().toLowerCase();
	if (!term) return options;

	return options.filter(
		(option) =>
			option.model.toLowerCase().includes(term) ||
			option.name?.toLowerCase().includes(term) ||
			option.aliases?.some((alias: string) => alias.toLowerCase().includes(term)) ||
			option.tags?.some((tag: string) => tag.toLowerCase().includes(term))
	);
}

export function groupModelOptions(
	filteredOptions: ModelOption[],
	favouriteIds: Set<string>,
	isModelLoaded: (model: string) => boolean
): GroupedModelOptions {
	// Loaded models
	const loaded: ModelItem[] = [];
	for (let i = 0; i < filteredOptions.length; i++) {
		if (isModelLoaded(filteredOptions[i].model)) {
			loaded.push({ option: filteredOptions[i], flatIndex: i });
		}
	}

	// Favourites (excluding loaded)
	const loadedModelIds = new Set(loaded.map((item) => item.option.model));
	const favourites: ModelItem[] = [];
	for (let i = 0; i < filteredOptions.length; i++) {
		if (
			favouriteIds.has(filteredOptions[i].model) &&
			!loadedModelIds.has(filteredOptions[i].model)
		) {
			favourites.push({ option: filteredOptions[i], flatIndex: i });
		}
	}

	// Available models grouped by org (excluding loaded and favourites)
	const available: OrgGroup[] = [];
	const orgGroups = new SvelteMap<string, ModelItem[]>();
	for (let i = 0; i < filteredOptions.length; i++) {
		const option = filteredOptions[i];
		if (loadedModelIds.has(option.model) || favouriteIds.has(option.model)) continue;

		const key = option.parsedId?.orgName ?? '';
		if (!orgGroups.has(key)) orgGroups.set(key, []);
		orgGroups.get(key)!.push({ option, flatIndex: i });
	}

	for (const [orgName, items] of orgGroups) {
		available.push({ orgName: orgName || null, items });
	}

	return { loaded, favourites, available };
}
