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
	limit = Infinity
): ProviderGroup[] {
	const byBackend = new SvelteMap<string, ModelItem[]>();

	for (let i = 0; i < options.length; i++) {
		const option = options[i];
		const backendId = option.backendId ?? '';

		if (!byBackend.has(backendId)) byBackend.set(backendId, []);

		byBackend.get(backendId)!.push({ flatIndex: i, option });
	}

	return providers.map((provider) => {
		const items = byBackend.get(provider.backendId) ?? [];

		return { ...provider, items: items.slice(0, limit), matched: items.length };
	});
}
