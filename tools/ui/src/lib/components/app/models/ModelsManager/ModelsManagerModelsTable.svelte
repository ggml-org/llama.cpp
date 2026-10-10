<script lang="ts">
	import ModelsManagerModelRow from './ModelsManagerModelRow.svelte';
	import ModelsManagerQuantRow from './ModelsManagerQuantRow.svelte';
	import ModelsManagerRepoRow from './ModelsManagerRepoRow.svelte';
	import ModelsManagerTableToolbar from './ModelsManagerTableToolbar.svelte';
	import {
		hasActiveFilters,
		modelContextLength,
		type ModelQuantGroup,
		type ModelsTableGroup,
		statusRank
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
		CollapsibleRegion,
		GroupedList,
		type GroupedListGroup,
		GroupedListUnit,
		Logo,
		ModelAvatar,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import {
		FAMILY_ROW_WINDOW,
		type ModalityKey,
		MODEL_ROW_GRID_CLASS,
		MODEL_ROW_TRAILING_CELL_CLASS,
		MODEL_ROW_WINDOW,
		MODELS_TABLE_GROUP_STICKY_OFFSET,
		SETTINGS_KEYS
	} from '$lib/constants';
	import {
		KeyboardKey,
		ModelCapability,
		ModelDownloadConfirmAction,
		ModelsTableGroupKind,
		ModelsTableSortKey
	} from '$lib/enums';
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
	let hasFilters = $derived(hasActiveFilters(contextLimit, modalities, capabilities));

	/** Noun the show-more row counts in, per unit of the grouped list. */
	const SHOW_MORE_NOUNS: Record<GroupedListUnit, string> = {
		[GroupedListUnit.ENTRIES]: 'models',
		[GroupedListUnit.GROUPS]: 'families'
	};
	let pendingDelete = $state('');
	let deleteOpen = $state(false);
	/** Repos whose quants are folded away; the rest show them. */
	const collapsedQuants = new SvelteSet<string>();
	/** Sections that list their models straight, without folding them into families. */
	const FLAT_SECTIONS = new Set<ModelsTableGroupKind>([
		ModelsTableGroupKind.DOWNLOADING,
		ModelsTableGroupKind.FAVORITES,
		ModelsTableGroupKind.LOADED
	]);
	let sections = $derived(
		groups.map((group) => {
			// a flat section lists its models straight, families or not
			const flat =
				FLAT_SECTIONS.has(group.kind) ||
				!settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY];
			// downloads keep the feed's order: their progress, not their name, moves
			const items =
				group.kind === ModelsTableGroupKind.DOWNLOADING ? group.items : sortEntries(group.items);

			return {
				...group,
				families: flat ? [] : groupModelFamilies(items, (entry) => entry.base.model),
				flat,
				items
			};
		})
	);

	function familyGroups(
		group: (typeof sections)[number]
	): GroupedListGroup<ModelFamilyGroup<ModelQuantGroup>, ModelQuantGroup>[] {
		return group.families.map((family) => ({
			entries: family.entries,
			group: family,
			key: `${group.key}::${family.key}`
		}));
	}

	// cancel is confirmed once for the whole list, so one dialog serves every row
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
				// a context we cannot read yet sorts with the smallest ones
				return (modelContextLength(a) ?? 0) - (modelContextLength(b) ?? 0);
			case ModelsTableSortKey.NAME:
				return a.model.localeCompare(b.model);
			case ModelsTableSortKey.STATUS:
				// a running model leads, then one that is being worked on (loading,
				// sleeping), then the rest; the reported status sorts the row's own cell
				return statusRank(b) - statusRank(a);
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
		if (event.key === KeyboardKey.SPACE) event.preventDefault();

		if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) toggle();
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

		<CollapsibleRegion open={!collapsedQuants.has(entry.key)}>
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
		</CollapsibleRegion>
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
		class="{MODEL_ROW_GRID_CLASS} relative group cursor-pointer rounded-md px-2 py-1 transition hover:bg-muted/40 max-md:px-3 max-md:py-2.5"
		onclick={toggle}
		onkeydown={(event) => handleFamilyKeydown(event, toggle)}
		role="button"
		tabindex="0"
	>
		<span class="flex min-w-0 items-center gap-3 max-md:pr-9">
			<ModelAvatar
				option={family.entries[0].base}
				showBaseModelAvatar
				showQuantBadge={false}
				size="size-6"
			/>

			<span class="truncate text-sm font-medium">{family.label}</span>

			<span class="text-sm text-muted-foreground">{countLabel}</span>
		</span>

		<!-- the context and status columns stay empty here: the fold control is this
		     row's last cell, which a phone floats at the row's end -->
		<span class="max-md:hidden"></span>

		<span class="max-md:hidden"></span>

		<span
			class="flex justify-center {MODEL_ROW_TRAILING_CELL_CLASS} {expanded
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
	unit: GroupedListUnit;
})}
	<div class="px-2 max-md:px-3">
		<button
			class="w-full cursor-pointer rounded-md px-2 py-2 text-left text-xs text-muted-foreground transition hover:bg-muted/40"
			onclick={onMore}
			type="button"
		>
			Show {count} more {SHOW_MORE_NOUNS[unit]}
		</button>
	</div>
{/snippet}

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
		class="{MODEL_ROW_GRID_CLASS} shrink-0 border-y border-border/40 px-2 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase max-md:px-3"
	>
		<span>{@render sortHeader(ModelsTableSortKey.NAME, 'Model')}</span>

		<span class="text-right whitespace-nowrap max-md:hidden">
			{@render sortHeader(ModelsTableSortKey.CONTEXT, 'Context')}
		</span>

		<span class="justify-self-center max-md:hidden">
			{@render sortHeader(ModelsTableSortKey.STATUS, 'Status')}
		</span>

		<span class="text-center max-md:hidden">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#each sections as group (group.key)}
			{#if group.items.length > 0}
				{#snippet groupIcon()}
					{#if group.kind === ModelsTableGroupKind.DOWNLOADING}
						<Download class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === ModelsTableGroupKind.FAVORITES}
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
					sectionHeaderClass="m-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground select-none max-md:px-3"
					sticky
					stickyClass="sticky z-10 bg-muted/90 backdrop-blur-lg"
				>
					<GroupedList
						group={familyRow}
						groupWindow={FAMILY_ROW_WINDOW}
						groups={group.flat ? null : familyGroups(group)}
						initialCollapsed={modelsStore.collapsedGroupsUnder(group.key)}
						item={listItem}
						items={group.flat ? group.items : []}
						keyOf={(entry) => entry.key}
						more={showMore}
						onCollapsedChange={(key, collapsed) =>
							modelsStore.setGroupCollapsed(group.key, key, collapsed)}
						sectionWindow={MODEL_ROW_WINDOW}
						stickyClass="bg-muted/30 backdrop-blur-lg"
						stickyStyle={MODELS_TABLE_GROUP_STICKY_OFFSET}
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
