<script lang="ts">
	import ModelsManagerModelSettings from './ModelsManagerModelSettings.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		isCustomized,
		isLocalOption,
		loadExtraArgs,
		loadOverrides,
		type ModelOverride,
		type ModelsTableGroup,
		saveOverrides
	} from './utils';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { backendsStore, conversationsStore, modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { formatFileSize } from '$lib/utils/formatters';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	interface Props {
		onClose?: () => void;
	}

	let { onClose }: Props = $props();

	let filter = $state('');
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

			return true;
		});
	});
	// loaded models lead the table, then favorites, then one block per provider;
	// a model is listed once, in the first group that claims it
	let groups = $derived.by(() => {
		// only llama-compat servers report a load state
		const isLlamaCompat = (option: ModelOption) =>
			getBackendCapabilities(getBackend(option.backendId)).loadUnload;
		const loaded = visible.filter(
			(option) => isLlamaCompat(option) && modelsStore.isModelLoaded(option.model)
		);
		const claimed = new SvelteSet(loaded.map((option) => option.id));
		const favorites = visible.filter(
			(option) => !claimed.has(option.id) && modelsStore.favoriteModelIds.has(option.model)
		);

		for (const option of favorites) claimed.add(option.id);

		const byBackend = new SvelteMap<string, ModelOption[]>();

		for (const option of visible) {
			if (claimed.has(option.id)) continue;

			const backendId = option.backendId ?? LOCAL_BACKEND_ID;

			if (!byBackend.has(backendId)) byBackend.set(backendId, []);

			byBackend.get(backendId)!.push(option);
		}

		const ordered: ModelsTableGroup[] = [];

		if (loaded.length) {
			ordered.push({
				backendId: null,
				isLocal: false,
				items: loaded,
				key: 'loaded',
				label: 'Loaded models'
			});
		}

		if (favorites.length) {
			ordered.push({
				backendId: null,
				isLocal: false,
				items: favorites,
				key: 'favorites',
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
				label: 'This server'
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
					label: backend.name
				});
			}
		}

		return ordered;
	});
	let matches = $derived(groups.flatMap((group) => group.items));

	let selected = $derived(allModels.find((option) => option.id === selectedId) ?? null);
	let summary = $derived.by(() => {
		const local = matches.filter(isLocalOption);
		const bytes = local.reduce((total, option) => {
			const size = option.meta?.size;

			return typeof size === 'number' ? total + size : total;
		}, 0);
		const loaded = matches.filter((option) => modelsStore.isModelLoaded(option.model)).length;

		return [
			`${local.length} local`,
			bytes > 0 ? `${formatFileSize(bytes)} on disk` : null,
			`${loaded} loaded`
		]
			.filter(Boolean)
			.join(' · ');
	});

	function toggleFavorite(option: ModelOption): void {
		modelsStore.toggleFavorite(option.model);
	}

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
		onClose?.();
	}

	function copyId(option: ModelOption): void {
		void navigator.clipboard.writeText(option.model).then(() => toast.success('Model id copied'));
	}

	function saveOverride(option: ModelOption, override: ModelOverride): void {
		overrides = { ...overrides, [option.id]: override };
		saveOverrides(overrides);
		toast.success(`Saved settings for ${option.name}`);
	}
</script>

<div
	class="grid min-h-0 flex-1"
	style="grid-template-columns: minmax(0, 1fr){selected ? ' 30rem' : ''};"
>
	<div class="min-h-0">
		<ModelsManagerModelsTable
			bind:filter
			{groups}
			{isFavorite}
			onCopyId={copyId}
			onSelect={(option) => (selectedId = option.id)}
			onToggleFavorite={toggleFavorite}
			onToggleLoad={(option) => void toggleLoad(option)}
			onUseInNewChat={(option) => void useInNewChat(option)}
			{selectedId}
			{summary}
		/>
	</div>

	{#if selected}
		<div class="min-h-0 overflow-hidden border-l border-border/40">
			<ModelsManagerModelSettings
				isCustomized={isCustomized(overrides[selected.id])}
				onClose={() => (selectedId = null)}
				onSave={(override) => saveOverride(selected, override)}
				onToggleLoad={() => void toggleLoad(selected)}
				onUseInNewChat={() => void useInNewChat(selected)}
				option={selected}
				override={overrides[selected.id]}
			/>
		</div>
	{/if}
</div>
