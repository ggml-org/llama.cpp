<script lang="ts">
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		groupModelQuants,
		type ModalityKey,
		modelContextLength,
		type ModelQuantGroup,
		MODELS_TABLE_GROUP_LABELS,
		type ModelsTableGroup,
		ModelsTableGroupKind,
		modelSupports
	} from './utils';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { type Snippet } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';

	interface Props {
		class?: string;
		/** Forwarded to the table's toolbar right end. */
		toolbarEnd?: Snippet;
	}

	let { class: className, toolbarEnd }: Props = $props();

	let filter = $state('');
	let contextLimit = $state(0);
	let modalityFilter = $state<ModalityKey[]>([]);
	let capabilityFilter = $state<ModelCapability[]>([]);
	let selectedId = $state<string | null>(null);

	let allModels = $derived(modelsStore.models);
	let isFavorite = $derived((option: ModelOption) =>
		modelsStore.favoriteModelIds.has(option.model)
	);
	let matching = $derived.by(() => {
		const term = filter.trim().toLowerCase();

		return allModels.filter((option) => {
			if (term && !`${option.name} ${option.model}`.toLowerCase().includes(term)) return false;

			// every capability asked for has to be there: tools and reasoning are
			// features a model either has or does not, unlike modalities
			if (
				capabilityFilter.length > 0 &&
				!capabilityFilter.every((capability) => modelSupports(option, capability))
			) {
				return false;
			}

			// a model whose modalities are unknown cannot be shown to match
			if (modalityFilter.length > 0 && !modalityFilter.some((key) => option.modalities?.[key])) {
				return false;
			}

			return contextLimit === 0 || modelContextLength(option) >= contextLimit;
		});
	});

	// recently used models lead their section, the rest keep the server's order;
	// derived, so a pick made while the manager is open reorders the sections
	let rank = $derived.by(() => {
		const map = new SvelteMap<string, number>();

		modelsStore.recentModelIds.forEach((id, index) => map.set(id, index));

		return map;
	});

	const rankOf = (entry: ModelQuantGroup) =>
		Math.min(...entry.quants.map((quant) => rank.get(quant.id) ?? Number.MAX_SAFE_INTEGER));
	const byRecency = (list: ModelQuantGroup[]) =>
		rank.size === 0 ? list : [...list].sort((a, b) => rankOf(a) - rankOf(b));
	// one entry per repo, so a model with several quants is a single table row;
	// loaded models lead the table, then favorites, then the local block
	let entries = $derived(byRecency(groupModelQuants(matching)));
	let groups = $derived.by(() => {
		// A loaded quant is a model of its own: it moves to the loaded section, and
		// the quants of its repo that are not loaded stay behind as that repo.
		const isLoaded = (option: ModelOption) => modelsStore.isModelLoaded(option.model);
		const loaded: ModelQuantGroup[] = [];
		const rest: ModelQuantGroup[] = [];

		for (const entry of entries) {
			const remaining = entry.quants.filter((quant) => !isLoaded(quant));

			for (const quant of entry.quants) {
				if (!isLoaded(quant)) continue;

				loaded.push({ ...entry, base: quant, key: quant.id, quants: [quant] });
			}

			if (remaining.length > 0) rest.push({ ...entry, base: remaining[0], quants: remaining });
		}

		const claimed = new SvelteSet<string>();
		const favorites = rest.filter((entry) =>
			entry.quants.some((q) => modelsStore.favoriteModelIds.has(q.model))
		);

		for (const entry of favorites) claimed.add(entry.key);

		const hidden = rest.filter(
			(entry) => !claimed.has(entry.key) && entry.quants.some((q) => modelsStore.isHidden(q.id))
		);
		const hiddenKeys = new SvelteSet(hidden.map((entry) => entry.key));
		const local = rest.filter((entry) => !claimed.has(entry.key) && !hiddenKeys.has(entry.key));
		const ordered: ModelsTableGroup[] = [];
		// loaded models lead the table, then favorites, then the local block
		const pushSection = (kind: ModelsTableGroupKind, items: ModelQuantGroup[]): void => {
			if (items.length === 0) return;

			ordered.push({
				items,
				key: kind === ModelsTableGroupKind.LOCAL ? LOCAL_BACKEND_ID : kind,
				kind,
				label: MODELS_TABLE_GROUP_LABELS[kind]
			});
		};

		pushSection(ModelsTableGroupKind.LOADED, loaded);
		pushSection(ModelsTableGroupKind.FAVORITES, favorites);
		pushSection(ModelsTableGroupKind.LOCAL, local);
		pushSection(ModelsTableGroupKind.HIDDEN, hidden);

		return ordered;
	});

	// a caller can ask for one model to be revealed, the download rows do
	$effect(() => {
		const focus = uiStore.manageModelFocus;

		if (!focus) return;

		const option = allModels.find((model) => model.id === focus || model.model === focus);

		if (option) selectedId = option.id;

		uiStore.manageModelFocus = null;
	});
</script>

<div class={['flex min-h-0 flex-1', className]}>
	<div class="min-h-0 min-w-0 flex-1">
		<ModelsManagerModelsTable
			bind:capabilities={capabilityFilter}
			bind:contextLimit
			bind:filter
			bind:modalities={modalityFilter}
			{groups}
			{isFavorite}
			onSelect={(option) => (selectedId = option.id)}
			{selectedId}
			{toolbarEnd}
		/>
	</div>
</div>
