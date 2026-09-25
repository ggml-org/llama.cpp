<script lang="ts">
	import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		groupModelQuants,
		isCustomized,
		loadExtraArgs,
		loadOverrides,
		type ModalityKey,
		modelContextLength,
		type ModelOverride,
		type ModelQuantGroup,
		type ModelsTableGroup,
		saveOverrides
	} from './utils';
	import { CollapsibleRegion } from '$lib/components/app';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { backendsStore, conversationsStore, modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { type Snippet, untrack } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	interface Props {
		class?: string;
		onClose?: () => void;
		/** Forwarded to the table's toolbar right end. */
		toolbarEnd?: Snippet;
	}

	let { class: className, onClose, toolbarEnd }: Props = $props();

	let filter = $state('');
	let providerFilter = $state<string[]>([]);
	let contextLimit = $state(0);
	let modalityFilter = $state<ModalityKey[]>([]);
	let selectedId = $state<string | null>(null);
	let overrides = $state<Record<string, ModelOverride>>(loadOverrides());

	let allModels = $derived(modelsStore.models);
	let isFavorite = $derived((option: ModelOption) =>
		modelsStore.favoriteModelIds.has(option.model)
	);
	// the rail counts follow the active view and filter, so it always says how many
	// models each provider contributes to what the table is showing
	let visible = $derived.by(() => {
		const term = filter.trim().toLowerCase();

		return allModels.filter((option) => {
			if (term && !`${option.name} ${option.model}`.toLowerCase().includes(term)) return false;

			if (providerFilter.length > 0) {
				const backendId = option.backendId ?? LOCAL_BACKEND_ID;

				if (!providerFilter.includes(backendId)) return false;
			}

			// a model whose modalities are unknown cannot be shown to match
			if (modalityFilter.length > 0 && !modalityFilter.some((key) => option.modalities?.[key])) {
				return false;
			}

			return contextLimit === 0 || modelContextLength(option) >= contextLimit;
		});
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
		// only llama-compat servers report a load state
		const isLlamaCompat = (entry: ModelQuantGroup) =>
			getBackendCapabilities(getBackend(entry.base.backendId)).loadUnload;
		const loaded = entries.filter(
			(entry) =>
				isLlamaCompat(entry) && entry.quants.some((q) => modelsStore.isModelLoaded(q.model))
		);
		const claimed = new SvelteSet(loaded.map((entry) => entry.key));
		const favorites = entries.filter(
			(entry) =>
				!claimed.has(entry.key) &&
				entry.quants.some((q) => modelsStore.favoriteModelIds.has(q.model))
		);

		for (const entry of favorites) claimed.add(entry.key);

		const hidden = entries.filter(
			(entry) => !claimed.has(entry.key) && entry.quants.some((q) => modelsStore.isHidden(q.id))
		);
		const hiddenKeys = new SvelteSet(hidden.map((entry) => entry.key));
		const byBackend = new SvelteMap<string, ModelQuantGroup[]>();

		for (const entry of entries) {
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
	let selected = $derived(allModels.find((option) => option.id === selectedId) ?? null);
	// the pane keeps a width of its own and stays in the DOM, so opening it slides a
	// fixed panel in rather than reflowing one into place. Each open remounts the
	// content, which is what reset the tabs when the pane used to unmount.
	let paneOption = $state<ModelOption | null>(null);
	let paneSession = $state(0);

	$effect(() => {
		const id = selectedId;

		if (!id) return;

		// untracked: this effect writes the session counter, and reading it back here
		// would make the effect invalidate itself
		untrack(() => {
			paneOption = allModels.find((option) => option.id === id) ?? null;
			paneSession += 1;
		});
	});

	// a caller can ask for one model to be revealed, the download rows do
	$effect(() => {
		const focus = uiStore.manageModelFocus;

		if (!focus) return;

		const option = allModels.find((model) => model.id === focus || model.model === focus);

		if (option) selectedId = option.id;

		uiStore.manageModelFocus = null;
	});

	async function toggleLoad(option: ModelOption): Promise<void> {
		if (modelsStore.isModelLoaded(option.model)) {
			await modelsStore.status.unload(option.model);

			return;
		}

		await modelsStore.status.load(option.model, loadExtraArgs(overrides[option.id]));
	}

	async function useInNewChat(option: ModelOption): Promise<void> {
		await modelsStore.selectModelById(option.id);
		await conversationsStore.openNewChat();
		// the chat is behind the dialog, so it takes focus once the dialog is out of the way
		uiStore.requestComposerFocus();
		onClose?.();
	}

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

{#snippet toolbarEndRegion()}
	<!-- the pane takes the toolbar's width, so the calls to action leave with it -->
	<CollapsibleRegion axis="width" open={!selected}>
		<div
			class="flex items-center gap-2 transition-opacity duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {selected
				? 'opacity-0'
				: 'opacity-100'}"
		>
			{@render toolbarEnd?.()}
		</div>
	</CollapsibleRegion>
{/snippet}

<div class={['flex min-h-0 flex-1', className]}>
	<div class="min-h-0 min-w-0 flex-1">
		<ModelsManagerModelsTable
			bind:contextLimit
			bind:filter
			bind:modalities={modalityFilter}
			bind:providers={providerFilter}
			{groups}
			{isFavorite}
			onSelect={(option) => (selectedId = option.id)}
			onUseAsDraft={useAsDraft}
			{overrides}
			{selectedId}
			toolbarEnd={toolbarEndRegion}
		/>
	</div>

	<div
		class="invisible w-0 shrink-0 overflow-clip transition-[width,visibility] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] data-[open=true]:visible data-[open=true]:w-[30rem]"
		data-open={selected !== null}
	>
		<!-- the content box keeps the open width, so it never reflows with the drawer -->
		<div
			class="flex h-full min-h-0 w-[30rem] max-w-[30rem] flex-col border-l border-border/40 transition-opacity duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {selected
				? 'opacity-100'
				: 'opacity-0'}"
		>
			{#if paneOption}
				{@const shown = paneOption}

				{#key paneSession}
					<ModelsManagerModelConfiguration
						isCustomized={isCustomized(overrides[shown.id])}
						onClose={() => (selectedId = null)}
						onSave={(override) => saveOverride(shown, override)}
						onToggleLoad={() => void toggleLoad(shown)}
						onUseInNewChat={() => void useInNewChat(shown)}
						option={shown}
						override={overrides[shown.id]}
					/>
				{/key}
			{/if}
		</div>
	</div>
</div>
