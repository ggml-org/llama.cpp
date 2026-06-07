import { ArrowDownToLine, Flame, Layers, Settings } from '@lucide/svelte';
import { SvelteMap } from 'svelte/reactivity';
import type { ModelOption } from '$lib/types/models';

export interface ModelItem {
	option: ModelOption;
	flatIndex: number;
}

export interface ModelLoadPhase {
	icon: typeof ArrowDownToLine;
	label: string;
	numeric: boolean; // true => phase reports 0..1 progress shown as a %
	anim: string; // animation class for indeterminate phases
}

// Map a router load stage (COMMON_LOAD_STAGE_* in common/common.h) to its presentation;
// shared by the dropdown trigger and the option rows so both render the same icon/label/%.
export function getModelLoadPhase(stage: string | null | undefined): ModelLoadPhase | null {
	switch (stage) {
		case 'download':
			return { icon: ArrowDownToLine, label: 'Downloading', numeric: true, anim: '' };
		case 'load':
			return { icon: Layers, label: 'Loading weights', numeric: true, anim: '' };
		case 'warmup':
			return { icon: Flame, label: 'Warming up', numeric: false, anim: 'animate-pulse' };
		case 'finalize':
			return { icon: Settings, label: 'Finalizing', numeric: false, anim: 'animate-spin' };
		default:
			return null;
	}
}

export interface OrgGroup {
	orgName: string | null;
	items: ModelItem[];
}

export interface GroupedModelOptions {
	loaded: ModelItem[];
	favorites: ModelItem[];
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
	favoriteIds: Set<string>,
	isModelLoaded: (model: string) => boolean
): GroupedModelOptions {
	// Loaded models
	const loaded: ModelItem[] = [];
	for (let i = 0; i < filteredOptions.length; i++) {
		if (isModelLoaded(filteredOptions[i].model)) {
			loaded.push({ option: filteredOptions[i], flatIndex: i });
		}
	}

	// Favorites (excluding loaded)
	const loadedModelIds = new Set(loaded.map((item) => item.option.model));
	const favorites: ModelItem[] = [];
	for (let i = 0; i < filteredOptions.length; i++) {
		if (
			favoriteIds.has(filteredOptions[i].model) &&
			!loadedModelIds.has(filteredOptions[i].model)
		) {
			favorites.push({ option: filteredOptions[i], flatIndex: i });
		}
	}

	// Available models grouped by org (excluding loaded and favorites)
	const available: OrgGroup[] = [];
	const orgGroups = new SvelteMap<string, ModelItem[]>();
	for (let i = 0; i < filteredOptions.length; i++) {
		const option = filteredOptions[i];
		if (loadedModelIds.has(option.model) || favoriteIds.has(option.model)) continue;

		const key = option.parsedId?.orgName ?? '';
		if (!orgGroups.has(key)) orgGroups.set(key, []);
		orgGroups.get(key)!.push({ option, flatIndex: i });
	}

	for (const [orgName, items] of orgGroups) {
		available.push({ orgName: orgName || null, items });
	}

	return { loaded, favorites, available };
}
