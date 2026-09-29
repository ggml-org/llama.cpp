import { ModelModality } from '$lib/enums';
import type { ModelOption } from '$lib/types/models';
import { SvelteMap } from 'svelte/reactivity';

export interface ModelItem {
	option: ModelOption;
	flatIndex: number;
}

export interface OrgGroup {
	orgName: string | null;
	items: ModelItem[];
}

export interface GroupedModelOptions {
	available: OrgGroup[];
	loaded: ModelItem[];
}

function matchesModality(option: ModelOption, term: string): boolean {
	const modalities = option.modalities;

	if (!modalities) return false;

	switch (term) {
		case ModelModality.VISION.toLowerCase():
			return modalities.vision;
		case ModelModality.AUDIO.toLowerCase():
			return modalities.audio;
		case ModelModality.VIDEO.toLowerCase():
			return modalities.video;
		default:
			return false;
	}
}

export function filterModelOptions(options: ModelOption[], searchTerm: string): ModelOption[] {
	const term = searchTerm.trim().toLowerCase();

	if (!term) return options;

	return options.filter(
		(option) =>
			option.model.toLowerCase().includes(term) ||
			option.name?.toLowerCase().includes(term) ||
			option.aliases?.some((alias: string) => alias.toLowerCase().includes(term)) ||
			option.tags?.some((tag: string) => tag.toLowerCase().includes(term)) ||
			matchesModality(option, term)
	);
}

/**
 * Favorite models across every backend, in list order: the favorites tab spans
 * all backends, so they come from the full option list.
 */
export function groupFavoriteOptions(
	options: ModelOption[],
	favoriteIds: Set<string>
): ModelItem[] {
	const favorites: ModelItem[] = [];

	for (let i = 0; i < options.length; i++) {
		if (favoriteIds.has(options[i].model)) {
			favorites.push({ flatIndex: i, option: options[i] });
		}
	}

	return favorites;
}

/**
 * Cut the local groups down to a window of rows, loaded first, so the caller can
 * grow it as the list scrolls.
 */
export function windowLocalGroups(
	groups: GroupedModelOptions,
	limit: number
): { available: GroupedModelOptions['available']; loaded: ModelItem[]; shown: number } {
	const loaded = groups.loaded.slice(0, Math.max(0, limit));

	let budget = limit - loaded.length;

	const available: GroupedModelOptions['available'] = [];

	let shown = loaded.length;

	for (const group of groups.available) {
		if (budget <= 0) break;

		const items = group.items.slice(0, budget);

		budget -= items.length;
		shown += items.length;

		if (items.length > 0) available.push({ ...group, items });
	}

	return { available, loaded, shown };
}

export function groupModelOptions(
	filteredOptions: ModelOption[],
	isModelLoaded: (model: string) => boolean
): GroupedModelOptions {
	// Loaded models
	const loaded: ModelItem[] = [];

	for (let i = 0; i < filteredOptions.length; i++) {
		const option = filteredOptions[i];

		if (isModelLoaded(option.model)) {
			loaded.push({ flatIndex: i, option });
		}
	}

	const loadedModelIds = new Set(loaded.map((item) => item.option.model));
	const available: OrgGroup[] = [];
	const orgGroups = new SvelteMap<string, ModelItem[]>();

	for (let i = 0; i < filteredOptions.length; i++) {
		const option = filteredOptions[i];

		if (loadedModelIds.has(option.model)) continue;

		const key = option.parsedId?.orgName ?? '';

		if (!orgGroups.has(key)) orgGroups.set(key, []);

		orgGroups.get(key)!.push({ flatIndex: i, option });
	}

	for (const [orgName, items] of orgGroups) {
		available.push({ items, orgName: orgName || null });
	}

	return { available, loaded };
}
