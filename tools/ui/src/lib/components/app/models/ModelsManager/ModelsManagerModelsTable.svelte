<script lang="ts">
	import {
		isLocalOption,
		type ModelOverride,
		type ModelQuantGroup,
		type ModelsTableGroup
	} from './utils';
	import {
		ArrowDown,
		ArrowUp,
		Boxes,
		ChevronDown,
		ChevronUp,
		Download,
		Eye,
		EyeOff,
		Heart,
		HeartOff,
		MoreHorizontal,
		Power,
		Trash2
	} from '@lucide/svelte';
	import {
		DropdownMenuActions,
		GroupedList,
		type GroupedListGroup,
		Logo,
		ModelAvatar,
		ModelCapabilities,
		ModelContext,
		ModelId,
		ModelLoadControl,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { SearchInput } from '$lib/components/app/forms';
	import ModelsSelectorDownloadItem from '$lib/components/app/models/ModelsSelector/ModelsSelectorDownloadItem.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { FAMILY_ROW_WINDOW, MODEL_ROW_WINDOW } from '$lib/constants';
	import { ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore, settingsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { groupModelFamilies, type ModelFamilyGroup } from '$lib/utils/model-families';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		/** Per-model load and inference overrides, keyed by backend-qualified id. */
		overrides: Record<string, ModelOverride>;
		selectedId: string | null;
		summary: string;
	}

	let {
		filter = $bindable(''),
		groups,
		isFavorite,
		onSelect,
		overrides,
		selectedId,
		summary
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	let filterInput = $state<HTMLInputElement | null>(null);

	// the dialog hands focus to its first control, so the filter takes it instead
	$effect(() => {
		if (!uiStore.manageModelsOpen) return;

		let frames = 0;
		let handle = requestAnimationFrame(function focusFilter() {
			if (filterInput) {
				filterInput.focus({ preventScroll: true });

				return;
			}

			if (frames++ < 20) handle = requestAnimationFrame(focusFilter);
		});

		return () => cancelAnimationFrame(handle);
	});

	/** In-flight and paused downloads, tracked by the status feed. */
	let downloadEntries = $derived(modelsStore.status.getDownloadEntries());
	let pendingCancel = $state('');
	let cancelOpen = $state(false);
	let pendingDelete = $state('');
	let deleteOpen = $state(false);
	/** Repos whose quants are folded away; the rest show them. */
	const collapsedQuants = new SvelteSet<string>();
	/** Sections that list their models straight, without folding them into families. */
	const FLAT_SECTIONS = new Set<ModelsTableGroup['kind']>(['favorites', 'loaded']);
	let sections = $derived(
		groups.map((group) => {
			// a flat section lists its models straight, families or not
			const flat = FLAT_SECTIONS.has(group.kind) || !settingsStore.config.groupModelsByFamily;
			const items = sortEntries(group.items);

			return {
				...group,
				families: flat ? [] : groupModelFamilies(items, (entry) => entry.base.model),
				flat,
				items
			};
		})
	);

	/** Families of one section. */
	function familyGroups(
		group: (typeof sections)[number]
	): GroupedListGroup<ModelFamilyGroup<ModelQuantGroup>, ModelQuantGroup>[] {
		return group.families.map((family) => ({
			entries: family.entries,
			group: family,
			key: `${group.key}::${family.key}`
		}));
	}

	// Cancel is confirmed once for the whole list, so a single dialog instance
	// serves however many downloads are in flight.
	function requestCancel(repoWithTag: string): void {
		pendingCancel = repoWithTag;
		cancelOpen = true;
	}

	function requestDelete(option: ModelOption): void {
		pendingDelete = option.model;
		deleteOpen = true;
	}

	function toggleQuants(key: string): void {
		if (collapsedQuants.has(key)) {
			collapsedQuants.delete(key);
		} else {
			collapsedQuants.add(key);
		}
	}

	/** Row actions follow the app's dropdown pattern: icon, label, separators, variants. */
	function rowActions(option: ModelOption, canLoad: boolean, favorite: boolean, isHidden: boolean) {
		return [
			{
				icon: favorite ? HeartOff : Heart,
				label: favorite ? 'Remove from favorites' : 'Add to favorites',
				onclick: () => modelsStore.toggleFavorite(option.model)
			},
			...(canLoad
				? [
						{
							icon: Trash2,
							label: 'Delete from disk',
							onclick: () => requestDelete(option),
							separator: true,
							variant: 'destructive' as const
						}
					]
				: []),
			{
				icon: isHidden ? Eye : EyeOff,
				label: isHidden ? 'Unhide model' : 'Hide model',
				onclick: () => modelsStore.toggleHidden(option.id),
				separator: true
			}
		];
	}
	const rowGrid = 'grid grid-cols-[minmax(0,1fr)_11rem_3rem_4.5rem] items-center gap-4';

	function stateOf(option: ModelOption): ServerModelStatus | null {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	}

	/** Context the model runs with: the stored override, else what a loaded model reports. */
	function configuredContext(option: ModelOption): number | null {
		const override = overrides[option.id]?.load?.contextLength;

		if (override) return override;

		return isLocalOption(option) && isLoadedOption(option)
			? modelsStore.props.getModelContextSize(option.model)
			: null;
	}

	type SortKey = 'context' | 'name' | 'status';

	/** Column the table is ordered by; unset keeps the manager's own order. */
	let sortKey = $state<SortKey | null>(null);
	let sortAsc = $state(true);

	/** The context column's own value: what the listing reports, else what the Hub does. */
	function contextOf(option: ModelOption): number {
		return (
			option.contextLength ??
			HuggingFaceService.cachedDetails(option.model)?.gguf?.context_length ??
			0
		);
	}

	function compareEntries(left: ModelQuantGroup, right: ModelQuantGroup): number {
		const a = left.base;
		const b = right.base;

		switch (sortKey) {
			case 'context':
				return contextOf(a) - contextOf(b);
			case 'name':
				return a.model.localeCompare(b.model);
			case 'status':
				return Number(isLoadedOption(b)) - Number(isLoadedOption(a));
			default:
				return 0;
		}
	}

	function sortEntries(entries: ModelQuantGroup[]): ModelQuantGroup[] {
		if (!sortKey) return entries;

		const direction = sortAsc ? 1 : -1;

		return [...entries].sort((a, b) => direction * compareEntries(a, b));
	}

	/** A click on a new column sorts lowest first, then highest first, then clears. */
	function toggleSort(key: SortKey): void {
		if (sortKey !== key) {
			sortKey = key;
			sortAsc = true;

			return;
		}

		if (sortAsc) {
			sortAsc = false;

			return;
		}

		sortKey = null;
	}

	function sortTitle(key: SortKey, label: string): string {
		const name = label.toLowerCase();

		if (sortKey !== key) return `Sort by ${name}, lowest first`;

		return sortAsc ? `Sort by ${name}, highest first` : `Stop sorting by ${name}`;
	}

	function isLoadedOption(option: ModelOption): boolean {
		const status = stateOf(option);

		return (
			(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
			!modelsStore.status.isOperationInProgress(option.model)
		);
	}
</script>

{#snippet statusDot(option: ModelOption)}
	{@const status = stateOf(option)}
	{@const isOperationInProgress = modelsStore.status.isOperationInProgress(option.model)}
	{@const isLoading = status === ServerModelStatus.LOADING || isOperationInProgress}
	{@const isFailed = status === ServerModelStatus.FAILED}
	{@const isSleeping = status === ServerModelStatus.SLEEPING}

	<ModelLoadControl
		canLoad={getBackendCapabilities(getBackend(option.backendId)).loadUnload}
		class="justify-self-center"
		{isFailed}
		isLoaded={isLoadedOption(option)}
		{isLoading}
		{isSleeping}
		{option}
		showRemoteMark
	/>
{/snippet}

{#snippet sortHeader(key: SortKey, label: string)}
	<button
		class="inline-flex cursor-pointer items-center gap-1 uppercase transition hover:text-foreground focus:outline-none"
		onclick={() => toggleSort(key)}
		title={sortTitle(key, label)}
		type="button"
	>
		{label}

		{#if sortKey === key}
			{#if sortAsc}
				<ArrowUp class="h-3 w-3" />
			{:else}
				<ArrowDown class="h-3 w-3" />
			{/if}
		{/if}
	</button>
{/snippet}

{#snippet row(option: ModelOption, indent = 0)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isHidden = modelsStore.isHidden(option.id)}

	<!-- <div class="px-2"> -->
	<div
		class={[
			rowGrid,
			'group cursor-pointer rounded-md px-2 py-3 transition',
			isHidden && 'opacity-60',
			selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
		]}
		onclick={() => onSelect(option)}
		onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
		role="button"
		tabindex="0"
	>
		<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
			<ModelAvatar
				{option}
				showBaseModelAvatar={!settingsStore.config.groupModelsByFamily}
				showRepoOrgAvatar={settingsStore.config.groupModelsByFamily}
				size="size-9"
			/>

			<ModelId
				aliases={option.aliases}
				class="min-w-0 flex-1"
				hideCapabilities
				modalities={option.modalities}
				modelId={option.model}
				tags={option.tags}
				title={option.model}
			/>

			<ModelCapabilities {option} />
		</span>

		<ModelContext class="justify-self-end" configured={configuredContext(option)} {option} />

		{@render statusDot(option)}

		<div class="flex items-center justify-center justify-self-center">
			<DropdownMenuActions
				actions={rowActions(option, canLoad, favorite, isHidden)}
				align="end"
				triggerIcon={MoreHorizontal}
				triggerTooltip="Model actions"
			/>
		</div>
	</div>
	<!-- </div> -->
{/snippet}

{#snippet repoRow(entry: ModelQuantGroup, indent = 0)}
	{@const isExpanded = !collapsedQuants.has(entry.key)}
	{@const providerCount = new Set(entry.quants.map((option) => option.backendId ?? '')).size}
	{@const groupLabel =
		entry.kind === 'providers'
			? `${providerCount} provider${providerCount === 1 ? '' : 's'}`
			: entry.kind === 'variants'
				? `${entry.quants.length} variants`
				: `${entry.quants.length} quants available`}
	{@const anyLoaded = entry.quants.some(isLoadedOption)}
	// a repo row stands for its quants, so it reports what they agree on
	{@const contextSource = entry.quants.find((quant) => quant.contextLength) ?? entry.base}
	{@const mediaSource = entry.quants.find((quant) => quant.modalities) ?? entry.base}

	<div
		class={[rowGrid, 'cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40']}
		onclick={() => toggleQuants(entry.key)}
		onkeydown={(event) => event.key === 'Enter' && toggleQuants(entry.key)}
		role="button"
		tabindex="0"
	>
		<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
			<ModelAvatar
				option={entry.base}
				showBaseModelAvatar={!settingsStore.config.groupModelsByFamily}
				showRepoOrgAvatar={settingsStore.config.groupModelsByFamily}
				size="size-9"
			/>

			<span class="min-w-0">
				<ModelId
					aliases={entry.base.aliases}
					class="min-w-0"
					hideCapabilities
					hideQuantization
					modalities={mediaSource.modalities}
					modelId={entry.base.model}
					tags={entry.base.tags}
					title={entry.base.model}
				/>

				<ModelCapabilities option={entry.base} />

				<span class="block text-xs text-muted-foreground">{groupLabel}</span>
			</span>
		</span>

		<ModelContext
			class="justify-self-end"
			configured={configuredContext(contextSource)}
			option={contextSource}
		/>

		<span class="justify-self-center">
			<span
				class="block h-2.5 w-2.5 rounded-full {anyLoaded
					? 'bg-emerald-500'
					: 'border border-muted-foreground/50'}"
			></span>
		</span>

		<span class="flex justify-center">
			{#if isExpanded}
				<ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
			{:else}
				<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
			{/if}
		</span>
	</div>
{/snippet}

{#snippet quantRow(option: ModelOption, indent = 0, showProvider = false)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isHidden = modelsStore.isHidden(option.id)}
	{@const quant = option.parsedId?.quantization ?? option.model}

	<div
		class={[
			rowGrid,
			'group cursor-pointer rounded-md px-2 py-2 transition',
			isHidden && 'opacity-60',
			selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
		]}
		onclick={() => onSelect(option)}
		onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
		role="button"
		tabindex="0"
	>
		<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
			<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">
				{showProvider ? (getBackend(option.backendId)?.name ?? quant) : quant}
			</Badge>

			<span class="truncate text-sm text-muted-foreground">{option.model}</span>
		</span>

		<ModelContext class="justify-self-end" configured={configuredContext(option)} {option} />

		{@render statusDot(option)}

		<div class="flex items-center justify-center justify-self-center">
			<DropdownMenuActions
				actions={rowActions(option, canLoad, favorite, isHidden)}
				align="end"
				triggerIcon={MoreHorizontal}
				triggerTooltip="Model actions"
			/>
		</div>
	</div>
{/snippet}

{#snippet entryTree(entry: ModelQuantGroup, indent = 0)}
	{#if entry.quants.length > 1}
		{@render repoRow(entry, indent)}

		{#if !collapsedQuants.has(entry.key)}
			{#each entry.quants as quant (quant.id)}
				{@render quantRow(quant, indent + 24, entry.kind === 'providers')}
			{/each}
		{/if}
	{:else}
		{@render row(entry.base, indent)}
	{/if}
{/snippet}

{#snippet familyRow({
	expanded,
	group: family,
	toggle
}: {
	expanded: boolean;
	group: ModelFamilyGroup<ModelQuantGroup>;
	toggle: () => void;
})}
	{@const countLabel = `${family.entries.length} model${family.entries.length === 1 ? '' : 's'}`}

	<!-- <div class="px-2"> -->
	<div
		class="{rowGrid} group cursor-pointer rounded-md px-2 py-1 transition hover:bg-muted/40"
		onclick={toggle}
		onkeydown={(event) => event.key === 'Enter' && toggle()}
		role="button"
		tabindex="0"
	>
		<span class="flex min-w-0 items-center gap-3">
			<ModelAvatar
				option={family.entries[0].base}
				showBaseModelAvatar
				showQuantBadge={false}
				size="size-6"
			/>

			<span class="truncate text-sm font-medium">{family.label}</span>

			<span class="text-sm text-muted-foreground">{countLabel}</span>
		</span>

		<span></span>

		<span></span>

		<span
			class="flex justify-center {expanded
				? 'opacity-0 group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100'
				: ''}"
		>
			{#if expanded}
				<ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
			{:else}
				<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
			{/if}
		</span>
	</div>
	<!-- </div> -->
{/snippet}

{#snippet listItem({ depth, entry }: { depth: number; entry: ModelQuantGroup })}
	{@render entryTree(entry, depth > 0 ? 16 : 0)}
{/snippet}

{#snippet showMore({
	count,
	onMore,
	unit
}: {
	count: number;
	onMore: () => void;
	unit: 'entries' | 'families';
})}
	<div class="px-2">
		<button
			class="w-full cursor-pointer rounded-md px-2 py-2 text-left text-xs text-muted-foreground transition hover:bg-muted/40"
			onclick={onMore}
			type="button"
		>
			Show {count} more {unit === 'families' ? 'families' : 'models'}
		</button>
	</div>
{/snippet}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.DELETE}
	onClose={() => (deleteOpen = false)}
	open={deleteOpen}
	repoWithTag={pendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 py-4">
		<SearchInput
			bind:ref={filterInput}
			bind:value={filter}
			class="max-w-64"
			placeholder="Filter models..."
		/>

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-2 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>{@render sortHeader('name', 'Model')}</span>

		<span class="text-right whitespace-nowrap">{@render sortHeader('context', 'Context')}</span>

		<span class="justify-self-center">{@render sortHeader('status', 'Status')}</span>

		<span class="text-center">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#if downloadEntries.length > 0}
			<ModelsSection count={downloadEntries.length} label="Download in progress" sticky>
				{#snippet icon()}
					<Download class="h-3.5 w-3.5 shrink-0" />
				{/snippet}

				{#each downloadEntries as entry (entry.repoWithTag)}
					<ModelsSelectorDownloadItem {entry} onRequestCancel={requestCancel} />
				{/each}
			</ModelsSection>
		{/if}

		{#each sections as group (group.key)}
			{#if group.items.length > 0}
				{#snippet groupIcon()}
					{#if group.kind === 'favorites'}
						<Heart class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'loaded'}
						<Power class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'hidden'}
						<EyeOff class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'local' || group.key === 'llama-compat'}
						<Logo class="shrink-0" style="--size: 0.875rem" />
					{:else if group.kind === 'compat'}
						<Boxes class="h-3.5 w-3.5 shrink-0" />
					{/if}
				{/snippet}

				<ModelsSection
					backendId={group.kind === 'provider' ? (group.backendId ?? undefined) : undefined}
					chevronClass="mr-7"
					count={group.items.length}
					defaultOpen={group.kind !== 'hidden'}
					icon={group.kind === 'provider' ? undefined : groupIcon}
					label={group.label}
					persistKey={group.key}
					revealChevronOnHover
					sticky
					stickyClass="sticky z-10 bg-muted/90 backdrop-blur-lg"
				>
					<GroupedList
						group={familyRow}
						groupStateKey={group.key}
						groupWindow={FAMILY_ROW_WINDOW}
						groups={group.flat ? null : familyGroups(group)}
						item={listItem}
						items={group.flat ? group.items : []}
						keyOf={(entry) => entry.key}
						more={showMore}
						sectionWindow={MODEL_ROW_WINDOW}
						stickyClass="bg-muted/30 backdrop-blur-lg"
						stickyStyle="top: calc(2.25rem - 1px)"
						weightOf={(entry) => entry.quants.length}
					/>
				</ModelsSection>
			{/if}
		{/each}

		{#if isEmpty}
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
