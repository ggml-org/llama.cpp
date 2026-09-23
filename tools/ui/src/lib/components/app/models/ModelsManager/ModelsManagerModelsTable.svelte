<script lang="ts">
	import type { ModelsTableGroup } from './utils';
	import { formatLastUsed, type ModelQuantGroup } from './utils';
	import {
		ChevronDown,
		ChevronRight,
		Eye,
		EyeOff,
		Heart,
		HeartOff,
		MoreHorizontal,
		Power,
		Trash2,
		Upload
	} from '@lucide/svelte';
	import {
		ActionIcon,
		DropdownMenuActions,
		GroupedList,
		type GroupedListGroup,
		Logo,
		ModelAvatar,
		ModelContext,
		ModelId,
		ModelLoadControl,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { FAMILY_ROW_WINDOW, MODEL_ROW_WINDOW } from '$lib/constants';
	import { ModelCapability, ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';
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
		onToggleLoad: (option: ModelOption) => void;
		selectedId: string | null;
		summary: string;
	}

	let {
		filter = $bindable(''),
		groups,
		isFavorite,
		onSelect,
		onToggleLoad,
		selectedId,
		summary
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
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

			return {
				...group,
				families: flat ? [] : groupModelFamilies(group.items, (entry) => entry.base.model),
				flat
			};
		})
	);

	/** Families of one section, the ones the user pinned leading it. */
	function familyGroups(
		group: (typeof sections)[number]
	): GroupedListGroup<ModelFamilyGroup<ModelQuantGroup>, ModelQuantGroup>[] {
		return group.families
			.map((family) => ({
				entries: family.entries,
				group: family,
				key: `${group.key}::${family.key}`
			}))
			.sort(
				(a, b) =>
					Number(modelsStore.isFavoriteFamily(b.key)) - Number(modelsStore.isFavoriteFamily(a.key))
			);
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
	function rowActions(
		option: ModelOption,
		canLoad: boolean,
		isLoaded: boolean,
		favorite: boolean,
		isHidden: boolean
	) {
		return [
			{
				icon: favorite ? HeartOff : Heart,
				label: favorite ? 'Remove from favorites' : 'Add to favorites',
				onclick: () => modelsStore.toggleFavorite(option.model)
			},
			...(canLoad
				? [
						{
							icon: isLoaded ? Upload : Power,
							label: isLoaded ? 'Unload model' : 'Load model',
							onclick: () => onToggleLoad(option)
						},
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
	const rowGrid = 'grid grid-cols-[minmax(0,1fr)_7rem_5rem_3rem_4.5rem] items-center gap-6';

	function stateOf(option: ModelOption): ServerModelStatus | null {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
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
	{@const isLoaded = isLoadedOption(option)}
	{@const isFailed = status === ServerModelStatus.FAILED}
	{@const isSleeping = status === ServerModelStatus.SLEEPING}

	<ModelLoadControl
		canLoad={getBackendCapabilities(getBackend(option.backendId)).loadUnload}
		class="justify-self-center"
		{isFailed}
		{isLoaded}
		{isLoading}
		{isSleeping}
		{option}
		showAction={false}
		showRemoteMark
	/>
{/snippet}

{#snippet row(option: ModelOption, indent = 0)}
	{@const isLoaded = isLoadedOption(option)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isHidden = modelsStore.isHidden(option.id)}

	<!-- <div class="px-2"> -->
	<div
		class={[
			rowGrid,
			'cursor-pointer rounded-md px-2 py-2.5 transition',
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
				modalities={option.modalities}
				modelId={option.model}
				supportsThinking={option.capabilities.includes(ModelCapability.REASONING)}
				supportsToolUse={option.capabilities.includes(ModelCapability.TOOL_USE)}
				tags={option.tags}
				title={option.model}
			/>
		</span>

		<ModelContext class="justify-self-end" {option} />

		<span class="justify-self-end text-sm text-muted-foreground">
			{formatLastUsed(modelsStore.recentModelUsage[option.id])}
		</span>

		{@render statusDot(option)}

		<div class="flex items-center justify-center justify-self-center">
			<DropdownMenuActions
				actions={rowActions(option, canLoad, isLoaded, favorite, isHidden)}
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
	{@const groupLabel =
		entry.kind === 'variants'
			? `${entry.quants.length} variants`
			: `${entry.quants.length} quants available`}
	{@const anyLoaded = entry.quants.some(isLoadedOption)}

	<div class="px-2">
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
						hideQuantization
						modalities={entry.base.modalities}
						modelId={entry.base.model}
						supportsThinking={entry.base.capabilities.includes(ModelCapability.REASONING)}
						supportsToolUse={entry.base.capabilities.includes(ModelCapability.TOOL_USE)}
						tags={entry.base.tags}
						title={entry.base.model}
					/>

					<span class="block text-xs text-muted-foreground">{groupLabel}</span>
				</span>
			</span>

			<span></span>

			<span></span>

			<span class="justify-self-center">
				<span
					class="block h-2.5 w-2.5 rounded-full {anyLoaded
						? 'bg-emerald-500'
						: 'border border-muted-foreground/50'}"
				></span>
			</span>

			<span class="flex justify-center">
				{#if isExpanded}
					<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
				{:else}
					<ChevronRight class="h-3.5 w-3.5 text-muted-foreground" />
				{/if}
			</span>
		</div>
	</div>
{/snippet}

{#snippet quantRow(option: ModelOption, indent = 0)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isLoaded = isLoadedOption(option)}
	{@const isHidden = modelsStore.isHidden(option.id)}
	{@const quant = option.parsedId?.quantization ?? option.model}

	<div class="px-2">
		<div
			class={[
				rowGrid,
				'cursor-pointer rounded-md px-2 py-2 transition',
				isHidden && 'opacity-60',
				selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
			]}
			onclick={() => onSelect(option)}
			onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
			role="button"
			tabindex="0"
		>
			<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
				<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">{quant}</Badge>

				<span class="truncate text-sm text-muted-foreground">{option.model}</span>
			</span>

			<ModelContext class="justify-self-end" {option} />

			<span class="justify-self-end text-sm text-muted-foreground">
				{formatLastUsed(modelsStore.recentModelUsage[option.id])}
			</span>

			{@render statusDot(option)}

			<div class="flex items-center justify-center justify-self-center">
				<DropdownMenuActions
					actions={rowActions(option, canLoad, isLoaded, favorite, isHidden)}
					align="end"
					triggerIcon={MoreHorizontal}
					triggerTooltip="Model actions"
				/>
			</div>
		</div>
	</div>
{/snippet}

{#snippet entryTree(entry: ModelQuantGroup, indent = 0)}
	{#if entry.quants.length > 1}
		{@render repoRow(entry, indent)}

		{#if !collapsedQuants.has(entry.key)}
			{#each entry.quants as quant (quant.id)}
				{@render quantRow(quant, indent + 24)}
			{/each}
		{/if}
	{:else}
		{@render row(entry.base, indent)}
	{/if}
{/snippet}

{#snippet familyRow({
	expanded,
	group: family,
	key,
	toggle
}: {
	expanded: boolean;
	group: ModelFamilyGroup<ModelQuantGroup>;
	key: string;
	toggle: () => void;
})}
	{@const countLabel = `${family.entries.length} model${family.entries.length === 1 ? '' : 's'}`}
	{@const favorite = modelsStore.isFavoriteFamily(key)}

	<div class="px-2">
		<div
			class="{rowGrid} group cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40"
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
					size="size-7"
				/>

				<span class="truncate text-sm font-medium">{family.label}</span>

				<span class="text-sm text-muted-foreground">{countLabel}</span>

				<span
					class="flex shrink-0"
					onclick={(event) => event.stopPropagation()}
					onkeydown={(event) => event.stopPropagation()}
					role="presentation"
				>
					{#if favorite}
						<ActionIcon
							class="h-5 w-5 text-rose-500 hover:text-rose-400"
							icon={Heart}
							iconSize="h-4 w-4"
							onclick={() => modelsStore.toggleFamilyFavorite(key)}
							tooltip="Remove family from favorites"
							tooltipAsTitle
						/>
					{:else}
						<ActionIcon
							class="h-5 w-5 opacity-0 transition group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
							icon={Heart}
							iconSize="h-4 w-4"
							onclick={() => modelsStore.toggleFamilyFavorite(key)}
							tooltip="Add family to favorites"
							tooltipAsTitle
						/>
					{/if}
				</span>
			</span>

			<span></span>

			<span></span>

			<span></span>

			<span class="flex justify-center">
				{#if expanded}
					<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
				{:else}
					<ChevronRight class="h-3.5 w-3.5 text-muted-foreground" />
				{/if}
			</span>
		</div>
	</div>
{/snippet}

{#snippet listItem({ entry }: { entry: ModelQuantGroup })}
	{@render entryTree(entry, 16)}
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
	action={ModelDownloadConfirmAction.DELETE}
	onClose={() => (deleteOpen = false)}
	open={deleteOpen}
	repoWithTag={pendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 py-4">
		<Input bind:value={filter} class="h-8 max-w-64 text-sm" placeholder="Filter models..." />

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-2 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>Model</span>

		<span class="text-right">Context</span>

		<span class="text-right">Last used</span>

		<span class="text-center">Status</span>

		<span class="text-center">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#each sections as group (group.key)}
			{#if group.items.length > 0}
				{#snippet groupIcon()}
					{#if group.kind === 'favorites'}
						<Heart class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'loaded'}
						<Power class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'hidden'}
						<EyeOff class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'local'}
						<Logo class="shrink-0" style="--size: 0.875rem" />
					{/if}
				{/snippet}

				<ModelsSection
					backendId={group.kind === 'provider' ? (group.backendId ?? undefined) : undefined}
					count={group.items.length}
					icon={group.kind === 'provider' ? undefined : groupIcon}
					label={group.label}
					open={group.kind !== 'hidden'}
					sticky
				>
					<GroupedList
						group={familyRow}
						groupWindow={FAMILY_ROW_WINDOW}
						groups={group.flat ? null : familyGroups(group)}
						item={listItem}
						items={group.flat ? group.items : []}
						keyOf={(entry) => entry.key}
						more={showMore}
						sectionWindow={MODEL_ROW_WINDOW}
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
