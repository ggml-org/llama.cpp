<script lang="ts">
	import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
	import {
		downloadGroups,
		groupModelQuants,
		modelCapability,
		modelContextLength,
		type ModelQuantGroup,
		type ModelsTableGroup,
		splitHiddenQuants
	} from './utils';
	import { LOCAL_BACKEND_ID, type ModalityKey, MODELS_TABLE_GROUP_LABELS } from '$lib/constants';
	import { ModelCapability, ModelsTableGroupKind } from '$lib/enums';
	import { modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { filterModelOptions } from '$lib/utils';
	import type { Snippet } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';

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
	// the pane is the global model drawer, so the row highlight follows what it shows
	let selectedId = $derived(uiStore.modelDetailsFocus);

	// tracked downloads get their own section, listed like any other model
	let downloads = $derived(
		downloadGroups(modelsStore.status.getDownloadEntries(), modelsStore.models, (repo) =>
			modelsStore.status.getDraftSidecars(repo)
		)
	);

	let allModels = $derived(modelsStore.models);

	let isFavorite = $derived((option: ModelOption) =>
		modelsStore.favoriteModelIds.has(option.model)
	);
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
		// a tracked download stands in its own section, so it is not listed twice
		const isDownload = (option: ModelOption) =>
			modelsStore.status.isDownloadInProgress(option.model) ||
			modelsStore.status.isDownloadPaused(option.model);
		const loaded: ModelQuantGroup[] = [];
		const rest: ModelQuantGroup[] = [];

		for (const entry of entries) {
			const remaining = entry.quants.filter((quant) => !isLoaded(quant) && !isDownload(quant));

			for (const quant of entry.quants) {
				if (!isLoaded(quant)) continue;

				loaded.push({ ...entry, base: quant, key: quant.id, quants: [quant] });
			}

			if (remaining.length > 0) rest.push({ ...entry, base: remaining[0], quants: remaining });
		}

		// a favorite quant stands on its own, listed flat like a loaded one, and the
		// quants left behind stay with their repo in the local block
		const favorites: ModelQuantGroup[] = [];
		const localRest: ModelQuantGroup[] = [];

		for (const entry of rest) {
			for (const quant of entry.quants) {
				if (modelsStore.favoriteModelIds.has(quant.model)) {
					favorites.push({ ...entry, base: quant, key: quant.id, quants: [quant] });
				}
			}

			const remaining = entry.quants.filter(
				(quant) => !modelsStore.favoriteModelIds.has(quant.model)
			);

			if (remaining.length > 0) {
				localRest.push({ ...entry, base: remaining[0], quants: remaining });
			}
		}

		const { hidden, local } = splitHiddenQuants(localRest, (option) =>
			modelsStore.isHidden(option.id)
		);
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
		pushSection(ModelsTableGroupKind.DOWNLOADING, downloads);
		pushSection(ModelsTableGroupKind.FAVORITES, favorites);
		pushSection(ModelsTableGroupKind.LOCAL, local);
		pushSection(ModelsTableGroupKind.HIDDEN, hidden);

		return ordered;
	});

	/** Leave the models page for the chat it was opened from, with the composer focused. */
	function returnToChat(): void {
		uiStore.requestComposerFocus();
		void uiStore.closeModelsManager();
	}

	/** Overlays that handle their own Escape, so the page leaves them alone. */
	const OPEN_OVERLAY_SELECTOR = [
		'[role="dialog"][data-state="open"]',
		'[role="alertdialog"][data-state="open"]',
		'[role="menu"][data-state="open"]',
		'[role="listbox"][data-state="open"]'
	].join(', ');

	// Escape steps back the way the dialog's did: the pane's drawer closes itself, then
	// the next Escape leaves the page
	function handleKeydown(event: KeyboardEvent): void {
		if (event.key !== 'Escape' || event.defaultPrevented) return;

		if (document.querySelector(OPEN_OVERLAY_SELECTOR)) return;

		// a typed filter is not thrown away by a stray Escape
		if (event.target instanceof HTMLInputElement && event.target.value) return;

		returnToChat();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class={['relative flex min-h-0 flex-1', className]}>
	<div class="min-h-0 min-w-0 flex-1">
		<ModelsManagerModelsTable
			bind:capabilities={capabilityFilter}
			bind:contextLimit
			bind:filter
			bind:modalities={modalityFilter}
			{groups}
			{isFavorite}
			onSelect={(option) => uiStore.openModelDetails(option.id)}
			{selectedId}
			{toolbarEnd}
		/>
	</div>
</div>
