<script lang="ts">
	import ModelsManagerModelRow from './ModelsManagerModelRow.svelte';
	import ModelsManagerQuantRow from './ModelsManagerQuantRow.svelte';
	import ModelsManagerRepoRow from './ModelsManagerRepoRow.svelte';
	import ModelsManagerTableToolbar from './ModelsManagerTableToolbar.svelte';
	import {
		isModelRunning,
		type ModalityKey,
		modelContextLength,
		type ModelQuantGroup,
		type ModelsTableGroup,
		ModelsTableGroupKind,
		ModelsTableSortKey
	} from './utils';
	import {
		ArrowDown,
		ArrowUp,
		ChevronDown,
		ChevronUp,
		Download,
		EyeOff,
		Heart,
		Power
	} from '@lucide/svelte';
	import {
		GroupedList,
		type GroupedListGroup,
		Logo,
		ModelAvatar,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import ModelsSelectorDownloadItem from '$lib/components/app/models/ModelsSelector/ModelsSelectorDownloadItem.svelte';
	import { FAMILY_ROW_WINDOW, MODEL_ROW_GRID_CLASS, MODEL_ROW_WINDOW } from '$lib/constants';
	import { ModelCapability, ModelDownloadConfirmAction } from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { groupModelFamilies, type ModelFamilyGroup } from '$lib/utils/model-families';
	import type { Snippet } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
		selectedId: string | null;
		/** Rendered at the toolbar's right end, past the filters. */
		toolbarEnd?: Snippet;
	}

	let {
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		filter = $bindable(''),
		groups,
		isFavorite,
		modalities = $bindable<ModalityKey[]>([]),
		onSelect,
		selectedId,
		toolbarEnd
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	let hasFilters = $derived(contextLimit > 0 || modalities.length > 0 || capabilities.length > 0);

	/** In-flight and paused downloads, tracked by the status feed. */
	let downloadEntries = $derived(modelsStore.status.getDownloadEntries());
	let pendingCancel = $state('');
	let cancelOpen = $state(false);
	let pendingDelete = $state('');
	let deleteOpen = $state(false);
	/** Repos whose quants are folded away; the rest show them. */
	const collapsedQuants = new SvelteSet<string>();
	/** Sections that list their models straight, without folding them into families. */
	const FLAT_SECTIONS = new Set<ModelsTableGroupKind>([
		ModelsTableGroupKind.FAVORITES,
		ModelsTableGroupKind.LOADED
	]);
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

	/** Column the table is ordered by; unset keeps the manager's own order. */
	let sortKey = $state<ModelsTableSortKey | null>(null);
	let sortAsc = $state(true);

	function compareEntries(left: ModelQuantGroup, right: ModelQuantGroup): number {
		const a = left.base;
		const b = right.base;

		switch (sortKey) {
			case ModelsTableSortKey.CONTEXT:
				return modelContextLength(a) - modelContextLength(b);
			case ModelsTableSortKey.NAME:
				return a.model.localeCompare(b.model);
			case ModelsTableSortKey.STATUS:
				return Number(isModelRunning(b)) - Number(isModelRunning(a));
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
	function toggleSort(key: ModelsTableSortKey): void {
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

	function sortTitle(key: ModelsTableSortKey, label: string): string {
		const name = label.toLowerCase();

		if (sortKey !== key) return `Sort by ${name}, lowest first`;

		return sortAsc ? `Sort by ${name}, highest first` : `Stop sorting by ${name}`;
	}

	function handleFamilyKeydown(event: KeyboardEvent, toggle: () => void): void {
		if (event.key === ' ') event.preventDefault();

		if (event.key === 'Enter' || event.key === ' ') toggle();
	}
</script>

{#snippet sortHeader(key: ModelsTableSortKey, label: string)}
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

{#snippet entryTree(entry: ModelQuantGroup, indent = 0)}
	{#if entry.quants.length > 1}
		<ModelsManagerRepoRow
			{entry}
			expanded={!collapsedQuants.has(entry.key)}
			{indent}
			onToggle={() => toggleQuants(entry.key)}
		/>

		{#if !collapsedQuants.has(entry.key)}
			{#each entry.quants as quant (quant.id)}
				<ModelsManagerQuantRow
					indent={indent + 24}
					{isFavorite}
					onDelete={requestDelete}
					{onSelect}
					option={quant}
					selected={selectedId === quant.id}
				/>
			{/each}
		{/if}
	{:else}
		<ModelsManagerModelRow
			{indent}
			{isFavorite}
			onDelete={requestDelete}
			{onSelect}
			option={entry.base}
			selected={selectedId === entry.base.id}
		/>
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

	<div
		class="{MODEL_ROW_GRID_CLASS} group cursor-pointer rounded-md px-2 py-1 transition hover:bg-muted/40"
		onclick={toggle}
		onkeydown={(event) => handleFamilyKeydown(event, toggle)}
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
	<ModelsManagerTableToolbar
		bind:capabilities
		bind:contextLimit
		bind:filter
		bind:modalities
		{toolbarEnd}
	/>

	<div
		class="{MODEL_ROW_GRID_CLASS} shrink-0 border-y border-border/40 px-2 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>{@render sortHeader(ModelsTableSortKey.NAME, 'Model')}</span>

		<span class="text-right whitespace-nowrap">
			{@render sortHeader(ModelsTableSortKey.CONTEXT, 'Context')}
		</span>

		<span class="justify-self-center">
			{@render sortHeader(ModelsTableSortKey.STATUS, 'Status')}
		</span>

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
					{#if group.kind === ModelsTableGroupKind.FAVORITES}
						<Heart class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === ModelsTableGroupKind.LOADED}
						<Power class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === ModelsTableGroupKind.HIDDEN}
						<EyeOff class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === ModelsTableGroupKind.LOCAL}
						<Logo class="shrink-0" style="--size: 0.875rem" />
					{/if}
				{/snippet}

				<ModelsSection
					chevronClass="mr-7"
					count={group.items.length}
					defaultOpen={group.kind !== ModelsTableGroupKind.HIDDEN}
					icon={groupIcon}
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
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">
				{hasFilters ? 'No models match these filters.' : 'No models found.'}
			</p>
		{/if}
	</div>
</div>
