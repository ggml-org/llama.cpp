<script lang="ts">
	import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		groupModelQuants,
		modelContextLength,
		type ModelQuantGroup,
		type ModelsTableGroup,
		modelSupports
	} from './utils';
	import {
		LOCAL_BACKEND_ID,
		type ModalityKey,
		MODELS_TABLE_GROUP_LABELS,
		ModelsTableGroupKind
	} from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { conversationsStore, modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { type Snippet, untrack } from 'svelte';
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

	let selected = $derived(allModels.find((option) => option.id === selectedId) ?? null);
	// The pane is laid out before it is ever opened, so the first open only slides a
	// finished panel in. It renders the selection, else the model it last showed.
	let lastPicked = $state<ModelOption | null>(null);
	let target = $derived(selected ?? lastPicked ?? allModels[0] ?? null);
	let shownId = $state<string | null>(null);
	let isSwapping = $state(false);
	let fade = $state<'open' | 'swap'>('open');
	let shownOption = $derived(allModels.find((option) => option.id === shownId) ?? null);

	$effect(() => {
		const id = selectedId;

		if (!id) return;

		// untracked: the effect must not track the state it writes
		untrack(() => {
			lastPicked = allModels.find((option) => option.id === id) ?? null;
		});
	});

	// Another model fades the panel out, swaps it, then fades it back in.
	$effect(() => {
		const next = target?.id ?? null;

		if (!next) return;

		if (shownId === null) {
			untrack(() => (shownId = next));

			return;
		}

		if (next === shownId) {
			if (selected) {
				untrack(() => {
					isSwapping = false;
					fade = 'open';
				});
			}

			return;
		}

		untrack(() => {
			isSwapping = true;
			fade = 'swap';
		});

		const timer = setTimeout(() => {
			untrack(() => {
				shownId = next;
				isSwapping = false;
			});
		}, SWAP_FADE_MS);

		return () => clearTimeout(timer);
	});

	/** How long the panel takes to fade out before it swaps to another model. */
	const SWAP_FADE_MS = 120;
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

	// recently used models lead their section, the rest keep the server's order
	let rank = $derived.by(() => {
		const map = new SvelteMap<string, number>();

		modelsStore.recentModelIds.forEach((id, index) => map.set(id, index));

		return map;
	});

	const rankOf = (entry: ModelQuantGroup) =>
		Math.min(...entry.quants.map((quant) => rank.get(quant.id) ?? Number.MAX_SAFE_INTEGER));
	const byRecency = (list: ModelQuantGroup[]) =>
		rank.size === 0 ? list : [...list].sort((a, b) => rankOf(a) - rankOf(b));
	// one entry per repo, so a model with several quants takes a single table row
	let entries = $derived(byRecency(groupModelQuants(matching)));
	let groups = $derived.by(() => {
		// a loaded quant is a model of its own: its repo keeps the quants left behind
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
	async function toggleLoad(option: ModelOption): Promise<void> {
		if (modelsStore.isModelLoaded(option.model)) {
			await modelsStore.status.unload(option.model);

			return;
		}

		await modelsStore.status.load(option.model);
	}

	async function useInNewChat(option: ModelOption): Promise<void> {
		await modelsStore.selectModelById(option.id);
		await conversationsStore.openNewChat();
		// the chat is behind the dialog, so it takes focus once the dialog is out of the way
		uiStore.requestComposerFocus();
	}
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

	<div class="pane-drawer shrink-0" data-open={selected !== null}>
		<!-- the content box keeps the open width, so it never reflows with the drawer -->
		<div
			class="pane-content flex h-full min-h-0 w-[30rem] max-w-[30rem] flex-col border-l border-border/40"
			data-fade={fade}
			data-visible={selected !== null && !isSwapping}
		>
			{#if shownOption}
				{#key shownId}
					<ModelsManagerModelConfiguration
						onClose={() => (selectedId = null)}
						onToggleLoad={() => void toggleLoad(shownOption)}
						onUseInNewChat={() => void useInNewChat(shownOption)}
						option={shownOption}
					/>
				{/key}
			{/if}
		</div>
	</div>
</div>

<style>
	/*
	 * The drawer moves by width because the table behind it gets that space back, so
	 * the content box inside holds the open width and only the container changes.
	 * Opening takes the iOS-like drawer curve; closing is the system responding, so
	 * it snaps back on the stronger ease-out.
	 */
	.pane-drawer {
		width: 0;
		overflow: clip;
		visibility: hidden;
		transition:
			width 120ms cubic-bezier(0.23, 1, 0.32, 1),
			visibility 120ms;
	}

	.pane-drawer[data-open='true'] {
		width: 30rem;
		visibility: visible;
		transition:
			width 200ms cubic-bezier(0.32, 0.72, 0, 1),
			visibility 200ms;
	}

	.pane-content {
		opacity: 0;
		transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
	}

	.pane-content[data-visible='true'] {
		opacity: 1;
	}

	/* opening: the fade waits for the drawer to move */
	.pane-content[data-visible='true'][data-fade='open'] {
		transition: opacity 150ms cubic-bezier(0.23, 1, 0.32, 1) 80ms;
	}

	/* swapping models: out, then in, with no pause */
	.pane-content[data-visible='true'][data-fade='swap'] {
		transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
	}
</style>
