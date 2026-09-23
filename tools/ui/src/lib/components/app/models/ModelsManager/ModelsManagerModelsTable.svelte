<script lang="ts">
	import type { ModelsTableGroup } from './utils';
	import {
		formatLastUsed,
		groupModelFamilies,
		type ModelFamilyGroup,
		type ModelQuantGroup
	} from './utils';
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
		DropdownMenuActions,
		Logo,
		ModelAvatar,
		ModelContext,
		ModelId,
		ModelLoadControl,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Badge } from '$lib/components/ui/badge';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { ModelCapability, ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
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
	const collapsedFamilies = new SvelteSet<string>();
	let groupFamilies = $state(false);
	/** Sections with their repos folded into families, when that view is on. */
	let sections = $derived(
		groups.map((group) => {
			const families = groupModelFamilies(group.items);

			return {
				...group,
				families: groupFamilies ? families.filter((family) => family.entries.length > 1) : null,
				singles: groupFamilies
					? families
							.filter((family) => family.entries.length === 1)
							.flatMap((family) => family.entries)
					: []
			};
		})
	);

	function requestDelete(option: ModelOption): void {
		pendingDelete = option.model;
		deleteOpen = true;
	}

	function toggleFamily(key: string): void {
		if (collapsedFamilies.has(key)) {
			collapsedFamilies.delete(key);
		} else {
			collapsedFamilies.add(key);
		}
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

{#snippet row(option: ModelOption)}
	{@const isLoaded = isLoadedOption(option)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isHidden = modelsStore.isHidden(option.id)}

	<div class="px-2">
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
			<span class="flex min-w-0 items-center gap-3">
				<ModelAvatar {option} showBaseModelAvatar size="size-9" />

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
	</div>
{/snippet}

{#snippet repoRow(entry: ModelQuantGroup)}
	{@const isExpanded = !collapsedQuants.has(entry.key)}
	{@const anyLoaded = entry.quants.some(isLoadedOption)}

	<div class="px-2">
		<div
			class={[rowGrid, 'cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40']}
			onclick={() => toggleQuants(entry.key)}
			onkeydown={(event) => event.key === 'Enter' && toggleQuants(entry.key)}
			role="button"
			tabindex="0"
		>
			<span class="flex min-w-0 items-center gap-3">
				<ModelAvatar option={entry.base} showBaseModelAvatar size="size-9" />

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

					<span class="block text-xs text-muted-foreground">
						{entry.quants.length} quants available
					</span>
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

{#snippet quantRow(option: ModelOption)}
	{@const favorite = isFavorite(option)}
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isLoaded = isLoadedOption(option)}
	{@const isHidden = modelsStore.isHidden(option.id)}
	{@const quant = option.parsedId?.quantization ?? option.model}

	<div class="px-2">
		<div
			class={[
				rowGrid,
				'cursor-pointer rounded-md py-2 pr-2 pl-13 transition',
				isHidden && 'opacity-60',
				selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
			]}
			onclick={() => onSelect(option)}
			onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
			role="button"
			tabindex="0"
		>
			<span class="flex min-w-0 items-center gap-3">
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

{#snippet entryTree(entry: ModelQuantGroup)}
	{#if entry.quants.length > 1}
		{@render repoRow(entry)}

		{#if !collapsedQuants.has(entry.key)}
			{#each entry.quants as quant (quant.id)}
				{@render quantRow(quant)}
			{/each}
		{/if}
	{:else}
		{@render row(entry.base)}
	{/if}
{/snippet}

{#snippet familyRow(family: ModelFamilyGroup)}
	{@const isExpanded = !collapsedFamilies.has(family.key)}

	<div class="px-2">
		<div
			class="{rowGrid} cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40"
			onclick={() => toggleFamily(family.key)}
			onkeydown={(event) => event.key === 'Enter' && toggleFamily(family.key)}
			role="button"
			tabindex="0"
		>
			<span class="truncate text-sm font-medium">{family.label}</span>

			<span class="text-sm text-muted-foreground">{family.entries.length} models</span>

			<span></span>

			<span></span>

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

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.DELETE}
	onClose={() => (deleteOpen = false)}
	open={deleteOpen}
	repoWithTag={pendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 py-4">
		<Input bind:value={filter} class="h-8 max-w-64 text-sm" placeholder="Filter models..." />

		<span class="flex items-center gap-2">
			<Checkbox bind:checked={groupFamilies} id="group-families" />

			<Label class="cursor-pointer text-xs text-muted-foreground" for="group-families">
				Group families
			</Label>
		</span>

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
					{#if group.families}
						{#each group.families as family (family.key)}
							{@render familyRow(family)}

							{#if !collapsedFamilies.has(family.key)}
								{#each family.entries as entry (entry.key)}
									{@render entryTree(entry)}
								{/each}
							{/if}
						{/each}

						{#each group.singles as entry (entry.key)}
							{@render entryTree(entry)}
						{/each}
					{:else}
						{#each group.items as entry (entry.key)}
							{@render entryTree(entry)}
						{/each}
					{/if}
				</ModelsSection>
			{/if}
		{/each}

		{#if isEmpty}
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
