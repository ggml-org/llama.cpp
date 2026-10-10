import { ModelModality } from '$lib/enums';
import type { ModelOption } from '$lib/types/models';

/** Model list helpers: search, grouping and windowing of the selector rows. */

export interface ModelItem {
	option: ModelOption;
}

export interface OrgGroup {
	orgName: string | null;
	items: ModelItem[];
}

export interface GroupedModelOptions {
	available: OrgGroup[];
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
	return options.filter((option) => favoriteIds.has(option.model)).map((option) => ({ option }));
}

/**
 * Cut the local groups down to a window of rows, so the caller can grow it as
 * the list scrolls.
 */
export function windowLocalGroups(
	groups: GroupedModelOptions,
	limit: number
): { available: OrgGroup[]; shown: number } {
	let budget = limit;

	const available: OrgGroup[] = [];

	let shown = 0;

	for (const group of groups.available) {
		if (budget <= 0) break;

		const items = group.items.slice(0, budget);

		budget -= items.length;
		shown += items.length;

		if (items.length > 0) available.push({ ...group, items });
	}

	return { available, shown };
}

export function groupModelOptions(filteredOptions: ModelOption[]): GroupedModelOptions {
	const available: OrgGroup[] = [];
	const orgGroups = new Map<string, ModelItem[]>();

	for (const option of filteredOptions) {
		const key = option.parsedId?.orgName ?? '';

		if (!orgGroups.has(key)) orgGroups.set(key, []);

		orgGroups.get(key)!.push({ option });
	}

	for (const [orgName, items] of orgGroups) {
		available.push({ items, orgName: orgName || null });
	}

	return { available };
}
