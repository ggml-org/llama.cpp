<script lang="ts">
	import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		downloadGroups,
		groupModelQuants,
		isCustomized,
		loadExtraArgs,
		loadOverrides,
		modelCapability,
		modelContextLength,
		modelDraftBadges,
		type ModelOverride,
		type ModelQuantGroup,
		type ModelsTableGroup,
		saveOverrides,
		splitHiddenQuants
	} from './utils';
	import { LOCAL_BACKEND_ID, type ModalityKey, MODELS_TABLE_GROUP_LABELS } from '$lib/constants';
	import { ModelCapability, ModelsTableGroupKind, ModelsTableProviderKind } from '$lib/enums';
	import {
		backendsStore,
		conversationsStore,
		modelsStore,
		serverStore,
		uiStore
	} from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { filterModelOptions } from '$lib/utils';
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
		// the term matches what the selector search matches: name, model, aliases and tags
		const searched = filterModelOptions(allModels, filter);

		return searched.filter((option) => {
			// every capability asked for has to be there, but a model whose chat template
			// has not been read yet stays listed: the row fetches the record as it comes
			// near the viewport, so hiding it would keep it from ever answering
			if (
				capabilityFilter.length > 0 &&
				!capabilityFilter.every((capability) => modelCapability(option, capability) !== false)
			) {
				return false;
			}

			if (
				draftFilter &&
				modelDraftBadges(option, overrides[option.id]?.load?.speculativeDecoding).length === 0
			) {
				return false;
			}

			// a model whose modalities are unknown cannot be shown to match
			if (modalityFilter.length > 0 && !modalityFilter.some((key) => option.modalities?.[key])) {
				return false;
			}

			if (contextLimit === 0) return true;

			// the context of a row arrives with its Hub record, so a model we cannot read
			// yet stays listed; only a known context below the limit rules it out
			const context = modelContextLength(option);

			return context === null || context >= contextLimit;
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
	let rank = $derived.by(() => {
		const map = new SvelteMap<string, number>();

		modelsStore.recentModelIds.forEach((id, index) => map.set(id, index));

		return map;
	});

	const rankOf = (entry: ModelQuantGroup) =>
		Math.min(...entry.quants.map((quant) => rank.get(quant.id) ?? Number.MAX_SAFE_INTEGER));
	const byRecency = (list: ModelQuantGroup[]) =>
		rank.size === 0 ? list : [...list].sort((a, b) => rankOf(a) - rankOf(b));
	// one entry per repo, so a model with several quants takes a single table row;
	// loaded models lead the table, then favorites, then one block per provider
	let entries = $derived(byRecency(groupModelQuants(visible)));
	// tracked downloads get their own section, listed like any other model
	let downloads = $derived(
		downloadGroups(modelsStore.status.getDownloadEntries(), modelsStore.models)
	);
	let groups = $derived.by(() => {
		// a loaded quant is a model of its own: its repo keeps the quants left behind.
		// Only llama-compat servers report a load state.
		const isLoaded = (option: ModelOption) =>
			getBackendCapabilities(getBackend(option.backendId)).loadUnload &&
			modelsStore.isModelLoaded(option.model);
		// a tracked download stands in its own section, so it is not listed twice
		const isDownload = (option: ModelOption) =>
			modelsStore.status.isDownloadInProgress(option.model) ||
			modelsStore.status.isDownloadPaused(option.model);
		// a compat backend serves the selection itself: its selected model is a
		// section of its own, since it never reports a load state
		const isProviderSelected = (option: ModelOption) =>
			option.id === modelsStore.selectedModelId &&
			!getBackendCapabilities(getBackend(option.backendId)).loadUnload;
		const selected: ModelQuantGroup[] = [];
		const loaded: ModelQuantGroup[] = [];
		const rest: ModelQuantGroup[] = [];

		for (const entry of entries) {
			const remaining = entry.quants.filter(
				(quant) => !isLoaded(quant) && !isDownload(quant) && !isProviderSelected(quant)
			);

			for (const quant of entry.quants) {
				if (isProviderSelected(quant)) {
					selected.push({ ...entry, base: quant, key: quant.id, quants: [quant] });

					continue;
				}

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

		const { hidden, local } = splitHiddenQuants(
			rest.filter((entry) => !claimed.has(entry.key)),
			(option) => modelsStore.isHidden(option.id)
		);
		const byBackend = new SvelteMap<string, ModelQuantGroup[]>();

		for (const entry of local) {
			const backendId = entry.base.backendId ?? LOCAL_BACKEND_ID;

			if (!byBackend.has(backendId)) byBackend.set(backendId, []);

			byBackend.get(backendId)!.push(entry);
		}

		const ordered: ModelsTableGroup[] = [];
		// the selected compat model leads the table, then the loaded models, then
		// favorites, then one block per backend
		const pushSection = (
			kind: ModelsTableGroup['kind'],
			items: ModelQuantGroup[],
			provider?: { id: string; label: string }
		): void => {
			if (items.length === 0) return;

			const isLocal = kind === ModelsTableGroupKind.LOCAL;

			ordered.push({
				backendId: provider?.id ?? (isLocal ? LOCAL_BACKEND_ID : null),
				isLocal,
				items,
				key: provider?.id ?? (isLocal ? LOCAL_BACKEND_ID : kind),
				kind,
				label: provider?.label ?? MODELS_TABLE_GROUP_LABELS[kind as ModelsTableGroupKind]
			});
		};

		pushSection(ModelsTableGroupKind.SELECTED, selected);
		pushSection(ModelsTableGroupKind.LOADED, loaded);
		pushSection(ModelsTableGroupKind.DOWNLOADING, downloads);
		pushSection(ModelsTableGroupKind.FAVORITES, favorites);

		const localItems = byBackend.get(LOCAL_BACKEND_ID);

		if (localItems?.length) pushSection(ModelsTableGroupKind.LOCAL, localItems);

		for (const backend of backendsStore.enabled) {
			if (backend.id === LOCAL_BACKEND_ID) continue;

			const items = byBackend.get(backend.id);

			if (items?.length)
				pushSection(ModelsTableProviderKind.PROVIDER, items, {
					id: backend.id,
					label: backend.name
				});
		}

		pushSection(ModelsTableGroupKind.HIDDEN, hidden);

		return ordered;
	});
	let selected = $derived(allModels.find((option) => option.id === selectedId) ?? null);
	// The pane is laid out before it is ever opened, so the first open only slides a
	// finished panel in. It renders the selection, else the model it last showed, else
	// the first model in the list.
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

	// Another model fades the panel out, swaps it, then fades it back in. Reopening the
	// same one only fades it, so the panel keeps its tab.
	$effect(() => {
		const next = target?.id ?? null;
		const isOpen = selected !== null;

		if (!next) return;

		if (shownId === null) {
			untrack(() => (shownId = next));

			return;
		}

		if (next === shownId) {
			if (isOpen) {
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

	// a caller can ask for one model to be revealed, the download rows do
	$effect(() => {
		const focus = uiStore.manageModelFocus;

		if (!focus) return;

		const option = allModels.find((model) => model.id === focus || model.model === focus);

		if (option) selectedId = option.id;

		uiStore.manageModelFocus = null;
	});

	/** How long the panel takes to fade out before it swaps to another model. */
	const SWAP_FADE_MS = 120;
	/** How long the calls to action take to leave the toolbar, matching their fade. */
	const CTA_LEAVE_MS = 120;
	/** How long the panel takes to slide out before the calls to action come back. */
	const PANE_LEAVE_MS = 120;

	// The calls to action leave first, then the panel takes the space they gave up.
	// Closing runs the same order backwards.
	let ctasVisible = $state(true);
	// once faded the row leaves the flow, so the filters keep the room it was holding
	let ctasGone = $state(false);
	let paneOpen = $state(false);

	$effect(() => {
		if (selected !== null) {
			untrack(() => (ctasVisible = false));

			const timer = setTimeout(
				() =>
					untrack(() => {
						ctasGone = true;
						paneOpen = true;
					}),
				CTA_LEAVE_MS
			);

			return () => clearTimeout(timer);
		}

		untrack(() => {
			paneOpen = false;
			ctasGone = false;
		});

		const timer = setTimeout(() => untrack(() => (ctasVisible = true)), PANE_LEAVE_MS);

		return () => clearTimeout(timer);
	});

	async function toggleLoad(option: ModelOption): Promise<void> {
		if (modelsStore.isModelLoaded(option.model)) {
			await modelsStore.status.unload(option.model);

			return;
		}

		await modelsStore.status.load(option.model, loadExtraArgs(overrides[option.id]));
	}

	/** The chat sits behind the dialog, so the dialog closes and the composer takes focus. */
	function returnToChat(): void {
		uiStore.manageModelsOpen = false;
		uiStore.requestComposerFocus();
		onClose?.();
	}

	/** Load the model when the server can take load requests, without waiting for it. */
	function loadInBackground(option: ModelOption): void {
		// only the built-in server loads on request, and only in router mode
		if (serverStore.isRouterMode && !modelsStore.isModelLoaded(option.model)) {
			modelsStore.status
				.load(option.model)
				.catch((error) => console.error('Failed to load model:', error));
		}
	}

	/** Switch the open chat to this model, the way the desktop model dropdown does. */
	async function useInChat(option: ModelOption): Promise<void> {
		// a deliberate pick records the recency; useInNewChat skips it because a new
		// conversation is where the pick is exercised, not a return to a model the
		// user went looking for
		await modelsStore.selectModelById(option.id, { recordRecent: true });

		loadInBackground(option);

		returnToChat();
	}

	async function useInNewChat(option: ModelOption): Promise<void> {
		await modelsStore.selectModelById(option.id);
		await conversationsStore.openNewChat();

		loadInBackground(option);

		returnToChat();
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
	<!-- the calls to action fade in place; the pane waits for them to be gone -->
	<div
		class="transition-[opacity,visibility] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] {ctasGone
			? 'hidden'
			: 'flex items-center gap-2'} {ctasVisible ? 'visible opacity-100' : 'invisible opacity-0'}"
	>
		{@render toolbarEnd?.()}
	</div>
{/snippet}

<div class={['relative flex min-h-0 flex-1', className]}>
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
			toolbarEnd={toolbarEndRegion}
		/>
	</div>

	<!-- on a phone the pane covers the whole dialog: the manager header would only
	     repeat what the pane's own header says -->
	<div class="pane-drawer max-md:fixed max-md:inset-0 max-md:z-[60] shrink-0" data-open={paneOpen}>
		<!-- the content box keeps the open width, so it never reflows with the drawer -->
		<div
			class="pane-content flex h-full min-h-0 w-[30rem] max-w-[30rem] flex-col border-l border-border/40 max-md:w-full max-md:max-w-none max-md:border-l-0 max-md:bg-background"
			data-fade={fade}
			data-visible={paneOpen && !isSwapping}
		>
			{#if shownOption}
				{#key shownId}
					<ModelsManagerModelConfiguration
						isCustomized={isCustomized(overrides[shownOption.id])}
						onClose={() => (selectedId = null)}
						onSave={(override) => saveOverride(shownOption, override)}
						onToggleLoad={() => void toggleLoad(shownOption)}
						onUseInChat={() => void useInChat(shownOption)}
						onUseInNewChat={() => void useInNewChat(shownOption)}
						option={shownOption}
						override={overrides[shownOption.id]}
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

	/* a phone has no room beside the table: the pane covers it instead */
	@media (max-width: 767px) {
		.pane-drawer[data-open='true'] {
			width: auto;
		}
	}

	/* reduced motion keeps the fades and drops the slide */
	@media (prefers-reduced-motion: reduce) {
		.pane-drawer,
		.pane-drawer[data-open='true'] {
			transition: visibility 120ms;
		}

		.pane-content,
		.pane-content[data-visible='true'][data-fade='open'] {
			transition: opacity 100ms;
		}
	}
</style>
