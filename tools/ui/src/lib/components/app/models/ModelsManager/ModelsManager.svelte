<script lang="ts">
	import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		groupModelQuants,
		isCustomized,
		isLocalOption,
		loadExtraArgs,
		loadOverrides,
		type ModelOverride,
		type ModelQuantGroup,
		type ModelsTableGroup,
		saveOverrides
	} from './utils';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { ModelGroupingMode } from '$lib/enums/settings.enums';
	import {
		backendsStore,
		conversationsStore,
		modelsStore,
		settingsStore,
		uiStore
	} from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { formatFileSize } from '$lib/utils/formatters';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	interface Props {
		class?: string;
		onClose?: () => void;
	}

	let { class: className, onClose }: Props = $props();

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

		// by capability: one block whose models can load and unload, one that is chat only
		if (settingsStore.config.modelGrouping === ModelGroupingMode.COMPAT) {
			const local = (entry: ModelQuantGroup) => entry.base.backendId === LOCAL_BACKEND_ID;
			const loadable: ModelQuantGroup[] = [];
			const chatOnly: ModelQuantGroup[] = [];

			for (const entry of entries) {
				if (claimed.has(entry.key) || hiddenKeys.has(entry.key)) continue;

				const canLoad = getBackendCapabilities(getBackend(entry.base.backendId)).loadUnload;

				(canLoad ? loadable : chatOnly).push(entry);
			}

			// the bundled server leads the block it belongs to
			const leading = (list: ModelQuantGroup[]) =>
				[...list].sort((a, b) => Number(local(b)) - Number(local(a)));

			if (loadable.length) {
				ordered.push({
					backendId: null,
					isLocal: false,
					items: leading(loadable),
					key: 'llama-compat',
					kind: 'compat',
					label: 'Llama-compat'
				});
			}

			if (chatOnly.length) {
				ordered.push({
					backendId: null,
					isLocal: false,
					// one repo, one row per provider that serves it
					items: groupModelQuants(
						chatOnly.flatMap((entry) => entry.quants),
						true
					),
					key: 'oai-compat',
					kind: 'compat',
					label: 'OAI-compat'
				});
			}
		} else {
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
	let matches = $derived(entries.flatMap((entry) => entry.quants));

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

	function saveOverride(option: ModelOption, override: ModelOverride): void {
		overrides = { ...overrides, [option.id]: override };
		saveOverrides(overrides);
		toast.success(`Saved settings for ${option.name}`);
	}
</script>

<div
	class={['grid min-h-0 flex-1', className]}
	style="grid-template-columns: minmax(0, 1fr){selected ? ' 30rem' : ''};"
>
	<div class="min-h-0">
		<ModelsManagerModelsTable
			bind:filter
			{groups}
			{isFavorite}
			onSelect={(option) => (selectedId = option.id)}
			{overrides}
			{selectedId}
			{summary}
		/>
	</div>

	{#if selected}
		<div class="min-h-0 overflow-hidden border-l border-border/40">
			{#key selected.id}
				<ModelsManagerModelConfiguration
					isCustomized={isCustomized(overrides[selected.id])}
					onClose={() => (selectedId = null)}
					onSave={(override) => saveOverride(selected, override)}
					onToggleLoad={() => void toggleLoad(selected)}
					onUseInNewChat={() => void useInNewChat(selected)}
					option={selected}
					override={overrides[selected.id]}
				/>
			{/key}
		</div>
	{/if}
</div>
