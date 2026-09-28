<script lang="ts">
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		groupModelQuants,
		loadOverrides,
		type ModalityKey,
		modelContextLength,
		modelDraftsFor,
		type ModelOverride,
		type ModelQuantGroup,
		type ModelsTableGroup,
		modelSupports,
		saveOverrides
	} from './utils';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { backendsStore, modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { type Snippet } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	interface Props {
		class?: string;
		/** Forwarded to the table's toolbar right end. */
		toolbarEnd?: Snippet;
	}

	let { class: className, toolbarEnd }: Props = $props();

	let filter = $state('');
	let providerFilter = $state<string[]>([]);
	let contextLimit = $state(0);
	let modalityFilter = $state<ModalityKey[]>([]);
	let capabilityFilter = $state<ModelCapability[]>([]);
	let draftFilter = $state(false);
	let selectedId = $state<string | null>(null);
	let overrides = $state<Record<string, ModelOverride>>(loadOverrides());

	let allModels = $derived(modelsStore.models);
	let isFavorite = $derived((option: ModelOption) =>
		modelsStore.favoriteModelIds.has(option.model)
	);
	// every filter but the provider one, so a provider count does not fall to zero
	// the moment that provider is the one being looked at
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

			if (
				draftFilter &&
				modelDraftsFor(option, overrides[option.id]?.load?.speculativeDecoding).length === 0
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
	let visible = $derived.by(() =>
		providerFilter.length === 0
			? matching
			: matching.filter((option) => providerFilter.includes(option.backendId ?? LOCAL_BACKEND_ID))
	);
	// the rail counts follow the active view and filter, so it always says how many
	// repos each provider contributes to what the table is showing
	let providerCounts = $derived.by(() => {
		const counts: Record<string, number> = {};

		for (const entry of groupModelQuants(matching)) {
			const backendId = entry.base.backendId ?? LOCAL_BACKEND_ID;

			counts[backendId] = (counts[backendId] ?? 0) + 1;
		}

		return counts;
	});

	// recently used models lead their section, the rest keep the server's order
	const rank = new SvelteMap<string, number>();

	modelsStore.recentModelIds.forEach((id, index) => rank.set(id, index));

	const rankOf = (entry: ModelQuantGroup) =>
		Math.min(...entry.quants.map((quant) => rank.get(quant.id) ?? Number.MAX_SAFE_INTEGER));
	const byRecency = (list: ModelQuantGroup[]) =>
		rank.size === 0 ? list : [...list].sort((a, b) => rankOf(a) - rankOf(b));

	// one entry per repo, so a model with several quants is a single table row;
	// loaded models lead the table, then favorites, then one block per provider,
	// and an entry lands in the first group that claims it
	let entries = $derived(byRecency(groupModelQuants(visible)));
	let groups = $derived.by(() => {
		// A loaded quant is a model of its own: it moves to the loaded section, and
		// the quants of its repo that are not loaded stay behind as that repo. Only
		// llama-compat servers report a load state.
		const isLoaded = (option: ModelOption) =>
			getBackendCapabilities(getBackend(option.backendId)).loadUnload &&
			modelsStore.isModelLoaded(option.model);
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
		const byBackend = new SvelteMap<string, ModelQuantGroup[]>();

		for (const entry of rest) {
			if (claimed.has(entry.key) || hiddenKeys.has(entry.key)) continue;

			const backendId = entry.base.backendId ?? LOCAL_BACKEND_ID;

			if (!byBackend.has(backendId)) byBackend.set(backendId, []);

			byBackend.get(backendId)!.push(entry);
		}

		const ordered: ModelsTableGroup[] = [];

		if (loaded.length) {
			ordered.push({
				backendId: null,
				isLocal: false,
				items: loaded,
				key: 'loaded',
				kind: 'loaded',
				label: 'Loaded models'
			});
		}

		if (favorites.length) {
			ordered.push({
				backendId: null,
				isLocal: false,
				items: favorites,
				key: 'favorites',
				kind: 'favorites',
				label: 'Favorites'
			});
		}

		const localItems = byBackend.get(LOCAL_BACKEND_ID);

		if (localItems?.length) {
			ordered.push({
				backendId: LOCAL_BACKEND_ID,
				isLocal: true,
				items: localItems,
				key: LOCAL_BACKEND_ID,
				kind: 'local',
				label: 'Local models'
			});
		}

		for (const backend of backendsStore.enabled) {
			if (backend.id === LOCAL_BACKEND_ID) continue;

			const items = byBackend.get(backend.id);

			if (items?.length) {
				ordered.push({
					backendId: backend.id,
					isLocal: false,
					items,
					key: backend.id,
					kind: 'provider',
					label: backend.name
				});
			}
		}

		if (hidden.length) {
			ordered.push({
				backendId: null,
				isLocal: false,
				items: hidden,
				key: 'hidden',
				kind: 'hidden',
				label: 'Hidden models'
			});
		}

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

	/** Point the selected model's load settings at another model as its draft. */
	function useAsDraft(draft: ModelOption, targetId: string): void {
		const target = modelsStore.models.find((option) => option.id === targetId);

		if (!target) return;

		saveOverride(target, {
			...overrides[target.id],
			load: { ...overrides[target.id]?.load, speculativeDecoding: draft.id }
		});
	}

	function saveOverride(option: ModelOption, override: ModelOverride): void {
		overrides = { ...overrides, [option.id]: override };
		saveOverrides(overrides);
		toast.success(`Saved settings for ${option.name}`);
	}
</script>

<div class={['flex min-h-0 flex-1', className]}>
	<div class="min-h-0 min-w-0 flex-1">
		<ModelsManagerModelsTable
			bind:capabilities={capabilityFilter}
			bind:contextLimit
			bind:draft={draftFilter}
			bind:filter
			bind:modalities={modalityFilter}
			bind:providers={providerFilter}
			{groups}
			{isFavorite}
			onSelect={(option) => (selectedId = option.id)}
			onUseAsDraft={useAsDraft}
			{overrides}
			{providerCounts}
			{selectedId}
			{toolbarEnd}
		/>
	</div>
</div>
