<script lang="ts">
	import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import {
		DropdownMenuSearchable,
		ModelId,
		ModelsSelectorList,
		ModelsSelectorOption,
		ModelsSelectorTriggerIcon
	} from '$lib/components/app';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { MODEL_ICON } from '$lib/constants';
	import { KeyboardKey, ServerModelStatus } from '$lib/enums';
	import type { UseModelsSelectorReturn } from '$lib/hooks/use-models-selector.svelte';
	import { ModelsService } from '$lib/services/models.service';
	import { modelsStore, uiStore } from '$lib/stores';
	import type { ModelOption, ModelSidecarBadge } from '$lib/types/models';
	import type { ModelItem } from '$lib/utils';
	import { modelLoadFraction, repoOf } from '$lib/utils';

	interface Props {
		/** The selector state this shell renders; the owner derives it once. */
		ms: UseModelsSelectorReturn;
		currentModel?: string | null;
		disabled?: boolean;
		/** Model id the list highlights from the keyboard; the owner tracks it. */
		highlightedId?: string | null;
		/** Bind the menu's open state, so the owner sees the close too. */
		open?: boolean;
		showOrgName?: boolean;
		onHighlight?: (id: string | null) => void;
		onManageModels?: () => void;
		onModelKeyAction?: (modelId: string, unload: boolean) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
	}

	let {
		currentModel = null,
		disabled = false,
		highlightedId = null,
		ms,
		onHighlight,
		onManageModels,
		onModelKeyAction,
		onSearchKeyDown,
		open = $bindable(false),
		showOrgName = true
	}: Props = $props();

	const selectedOption = $derived(ms.getDisplayOption());
	const triggerModel = $derived(selectedOption?.model ?? null);

	let triggerStatus = $derived(
		triggerModel
			? modelsStore.routerModels.find((m) => m.id === triggerModel)?.status?.value
			: undefined
	);
	let triggerLoading = $derived(
		!!triggerModel &&
			(triggerStatus === ServerModelStatus.LOADING ||
				modelsStore.status.isOperationInProgress(triggerModel))
	);
	let triggerLoadPercent = $derived(
		triggerModel && triggerLoading
			? Math.round(modelLoadFraction(modelsStore.status.getLoadProgress(triggerModel)) * 100)
			: 0
	);

	/** Draft sidecar as it reads in the trigger tooltip, with its own quant. */
	function draftSidecarLabel(baseModel: string, badge: ModelSidecarBadge): string {
		const baseRepo = repoOf(baseModel);

		// a sidecar of the model's own repo reads as a bare tag, a foreign one keeps its id
		if (badge.repo === baseRepo) {
			return `${badge.kind.toUpperCase()}${badge.quant ? `:${badge.quant}` : ''}`;
		}

		return ModelsService.buildDownloadTag(badge.repo, badge.quant, badge.kind);
	}

	/** Raw id of the selected model, plus every draft sidecar it pulls. */
	function triggerTooltipLabel(option: ModelOption): string {
		const drafts = (option.draftSidecars ?? []).map((badge) =>
			draftSidecarLabel(option.model, badge)
		);

		return [option.model, ...drafts].join(' + ');
	}

	function handleManageModels() {
		open = false;

		// let the menu finish closing before the dialog takes focus
		setTimeout(() => (onManageModels ? onManageModels() : uiStore.openModelsManager()), 0);
	}
</script>

{#snippet selectorTriggerInner()}
	<ModelsSelectorTriggerIcon class="mr-0.375 size-3.5 shrink-0" option={selectedOption} />

	<span class="flex min-w-0 items-center gap-0.5">
		{#if selectedOption}
			<ModelId
				class="min-w-0 overflow-hidden"
				hideOrgName={!showOrgName}
				hideQuantization
				modelId={selectedOption.model}
			/>
		{:else}
			<span class="min-w-0 font-medium">Select model</span>
		{/if}
	</span>

	{#if ms.updating || ms.isLoadingModel || triggerLoading}
		<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
	{:else}
		<ChevronDown class="h-3 w-3.5 shrink-0" />
	{/if}

	{#if triggerLoading}
		<ModelLoadHighlight percent={triggerLoadPercent} />
	{/if}
{/snippet}

{#snippet modelOption(item: ModelItem, hideOrgName: boolean)}
	{@const { option } = item}
	{@const isSelected = currentModel === option.model || ms.activeId === option.id}
	{@const isHighlighted = option.id === highlightedId}
	{@const isFav = ms.isFavorite(option.model)}

	<ModelsSelectorOption
		{hideOrgName}
		{isFav}
		{isHighlighted}
		{isSelected}
		onKeyDown={(event) => {
			if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) {
				event.preventDefault();
				onModelKeyAction?.(option.id, event.altKey);
			}
		}}
		onMouseEnter={() => onHighlight?.(option.id)}
		onSelect={ms.handleSelect}
		{option}
	/>
{/snippet}

<DropdownMenu.Root bind:open onOpenChange={ms.handleOpenChange}>
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				<DropdownMenu.Trigger
					{...props}
					class={[
						`relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:h-8 max-md:px-2.25 max-md:py-1.25 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
						!ms.isCurrentModelInCache
							? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
							: 'text-foreground',
						open && 'text-foreground',
						'max-w-[min(calc(100vw-4rem) md:max-w-[min(calc(100cqw-9rem),25rem)]'
					]}
					disabled={disabled || ms.updating}
				>
					{@render selectorTriggerInner()}
				</DropdownMenu.Trigger>
			{/snippet}
		</Tooltip.Trigger>

		{#if selectedOption}
			<Tooltip.Content>
				<p class="font-mono">{triggerTooltipLabel(selectedOption)}</p>
			</Tooltip.Content>
		{/if}
	</Tooltip.Root>

	<DropdownMenu.Content
		align="end"
		class="w-full md:min-w-80 md:w-md max-w-[calc(100vw-2rem)] p-0! max-h-[min(40rem,calc(var(--bits-dropdown-menu-content-available-height)-1rem))]"
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<DropdownMenuSearchable
			emptyMessage={ms.emptyMessage}
			isEmpty={ms.isEmpty && ms.isCurrentModelInCache}
			onSearchChange={(v) => ms.setSearchTerm(v)}
			{onSearchKeyDown}
			placeholder="Search models..."
			searchClass="bg-transparent"
			searchValue={ms.searchTerm}
		>
			<!-- Option list; the search header sticks to the top and the actions
		     footer to the bottom of the content scrollport. -->
			{@render pickerContent('px-1.5')}

			{#snippet footer()}
				<DropdownMenu.Group class="px-2">
					<DropdownMenu.Item class="gap-2" onSelect={handleManageModels}>
						<MODEL_ICON class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

						Manage models
					</DropdownMenu.Item>
				</DropdownMenu.Group>
			{/snippet}
		</DropdownMenuSearchable>
	</DropdownMenu.Content>
</DropdownMenu.Root>

{#snippet pickerContent(listClass: string)}
	<div class={['models-list', listClass]}>
		{#if !ms.isCurrentModelInCache && currentModel}
			<!-- Show unavailable model as first option (disabled) -->
			<button
				aria-disabled="true"
				aria-selected="true"
				class="flex w-full cursor-not-allowed items-center bg-red-400/10 p-2 text-left text-sm text-red-400"
				disabled
				role="option"
				type="button"
			>
				<ModelId
					class="flex-1"
					hideOrgName={!showOrgName}
					hideQuantization
					modelId={currentModel}
				/>

				<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
			</button>
		{/if}

		{#if ms.isEmpty}
			<p class="px-4 py-3 text-sm text-muted-foreground">{ms.emptyMessage}</p>
		{/if}

		<ModelsSelectorList
			activeId={ms.activeId}
			{currentModel}
			favorites={ms.favoriteItems}
			groups={ms.groupedFilteredOptions}
			loaded={ms.loadedItems}
			onSelect={ms.handleSelect}
			renderOption={modelOption}
			sectionHeaderClass="[&:not(:first-child)]:mt-1 mb-1 px-2 py-2 text-[13px] font-semibold text-foreground/80 select-none"
			{showOrgName}
		/>
	</div>
{/snippet}
