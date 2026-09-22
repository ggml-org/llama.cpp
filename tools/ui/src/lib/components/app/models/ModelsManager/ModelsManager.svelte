<script lang="ts">
	import ModelsManagerModelSettings from './ModelsManagerModelSettings.svelte';
	import ModelsManagerModelsList from './ModelsManagerModelsList/ModelsManagerModelsList.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		isCustomized,
		isLocalOption,
		loadExtraArgs,
		loadOverrides,
		type ModelOverride,
		type ModelsListFilter,
		type ModelsProviderGroup,
		saveOverrides
	} from './utils';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { backendsStore, conversationsStore, modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { formatFileSize } from '$lib/utils/formatters';
	import { SvelteMap } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	interface Props {
		onClose?: () => void;
		onOpenDiscover?: () => void;
	}

	let { onClose, onOpenDiscover }: Props = $props();

	let filter = $state('');
	let view = $state<ModelsListFilter>('all');
	let selectedBackendId = $state<string | null>(null);
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

			if (view === 'favorites' && !modelsStore.favoriteModelIds.has(option.model)) return false;

			if (view === 'loaded' && !modelsStore.isModelLoaded(option.model)) return false;

			return true;
		});
	});
	let providers = $derived.by(() => {
		const counts = new SvelteMap<string, number>();

		for (const option of visible) {
			const backendId = option.backendId ?? LOCAL_BACKEND_ID;

			counts.set(backendId, (counts.get(backendId) ?? 0) + 1);
		}

		const ordered: ModelsProviderGroup[] = [];
		const localCount = counts.get(LOCAL_BACKEND_ID);

		if (localCount) {
			ordered.push({
				backendId: LOCAL_BACKEND_ID,
				count: localCount,
				isLocal: true,
				label: 'This server'
			});
		}

		for (const backend of backendsStore.enabled) {
			if (backend.id === LOCAL_BACKEND_ID) continue;

			const count = counts.get(backend.id);

			if (count) {
				ordered.push({
					backendId: backend.id,
					count,
					isLocal: false,
					label: backend.name
				});
			}
		}

		return ordered;
	});
	let matches = $derived(
		selectedBackendId === null
			? visible
			: visible.filter((option) => (option.backendId ?? LOCAL_BACKEND_ID) === selectedBackendId)
	);
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
	let title = $derived(
		view === 'favorites'
			? 'Favorites'
			: view === 'loaded'
				? 'Loaded models'
				: (providers.find((provider) => provider.backendId === selectedBackendId)?.label ??
					'All models')
	);

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
	style="grid-template-columns: 16rem minmax(0, 1fr){selected ? ' 30rem' : ''};"
>
	<div class="min-h-0 border-r border-border/40 px-2 py-3">
		<ModelsManagerModelsList
			onSelect={(backendId) => (selectedBackendId = backendId)}
			{providers}
			{selectedBackendId}
			totalCount={visible.length}
		/>
	</div>

	<div class="min-h-0">
		<ModelsManagerModelsTable
			bind:filter
			bind:view
			{isFavorite}
			models={matches}
			onCopyId={copyId}
			onGetModels={() => onOpenDiscover?.()}
			onSelect={(option) => (selectedId = option.id)}
			onToggleFavorite={toggleFavorite}
			onToggleLoad={(option) => void toggleLoad(option)}
			onUseInNewChat={(option) => void useInNewChat(option)}
			{selectedId}
			{summary}
			{title}
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
