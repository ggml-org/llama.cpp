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

/** One remote backend's section on the remote view. */
export interface ProviderGroup {
	backendId: string;
	/** Models the provider lists without any search filtering. */
	catalog: number;
	error: string | null;
	/** Rows to render, capped by the display limit. */
	items: ModelItem[];
	loading: boolean;
	/** Rows left after the search filter, before the cap. */
	matched: number;
	name: string;
}

export interface GroupedModelOptions {
	available: OrgGroup[];
	loaded: ModelItem[];
	/** Remote backends, one section each. */
	providers: ProviderGroup[];
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
 * Favorite models across every backend, in list order. The favorites tab spans
 * all backends, so they are collected from the full option list.
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
 * Cut the local groups down to a window of rows. Loaded models come first, then
 * the org groups, so the caller can grow the window as the list is scrolled.
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
	// Available models grouped by org (excluding loaded)
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

	return { available, loaded, providers: [] };
}

/**
 * Remote backends as sections, one per backend, in the given order. Each section
 * keeps at most `limit` rows; `matched` carries the full count so the caller can
 * offer the rest.
 */
export function groupProviderOptions(
	options: ModelOption[],
	providers: {
		backendId: string;
		catalog: number;
		error: string | null;
		loading: boolean;
		name: string;
	}[],
	limit = Infinity,
	recentIds: readonly string[] = []
): ProviderGroup[] {
	const byBackend = new SvelteMap<string, ModelItem[]>();
	const rank = new SvelteMap<string, number>();

	recentIds.forEach((id, index) => rank.set(id, index));

	const rankOf = (id: string) => rank.get(id) ?? Number.MAX_SAFE_INTEGER;
	// recently used models lead their section, the rest keep the backend's order
	const byRecency = (items: ModelItem[]) =>
		rank.size === 0 ? items : [...items].sort((a, b) => rankOf(a.option.id) - rankOf(b.option.id));

	for (let i = 0; i < options.length; i++) {
		const option = options[i];
		const backendId = option.backendId ?? '';

		if (!byBackend.has(backendId)) byBackend.set(backendId, []);

		byBackend.get(backendId)!.push({ flatIndex: i, option });
	}

	return providers.map((provider) => {
		const items = byRecency(byBackend.get(provider.backendId) ?? []);

		return { ...provider, items: items.slice(0, limit), matched: items.length };
	});
}
