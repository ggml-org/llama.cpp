<script lang="ts">
	import ModelLoadHighlight from './ModelLoadHighlight.svelte';
	import type { ModelItem } from './utils';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import {
		DialogModelInformation,
		DropdownMenuSearchable,
		ModelId,
		ModelsSelectorList,
		ModelsSelectorOption
	} from '$lib/components/app';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { MODEL_SELECTOR_ICON } from '$lib/constants';
	import { KeyboardKey, ServerModelStatus } from '$lib/enums';
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';
	import { modelsStore } from '$lib/stores';
	import { modelLoadFraction } from '$lib/utils';

	interface Props {
		class?: string;
		currentModel?: string | null;
		disabled?: boolean;
		forceForegroundText?: boolean;
		onModelChange?: (modelId: string, modelName: string) => Promise<boolean> | boolean | void;
		useGlobalSelection?: boolean;
	}

	let {
		class: className = '',
		currentModel = null,
		disabled = false,
		forceForegroundText = false,
		onModelChange,
		useGlobalSelection = false
	}: Props = $props();

	let isOpen = $state(false);
	let highlightedId = $state<string | null>(null);

	const ms = useModelsSelector({
		currentModel: () => currentModel,
		onModelChange: () => onModelChange,
		onOpenChange: (open) => {
			isOpen = open;
			highlightedId = null;
		},
		useGlobalSelection: () => useGlobalSelection
	});

	$effect(() => {
		void ms.searchTerm;
		highlightedId = null;
	});

	// Focus the dropdown's search box without scrolling the page. bits-ui
	// auto-focuses the opened content by default, which can yank the page
	// scroll; we prevent that on the Content and refocus the search here.
	$effect(() => {
		if (!isOpen) return;

		requestAnimationFrame(() => {
			const search = document.querySelector<HTMLElement>(
				'[data-slot="dropdown-menu-content"] input'
			);

			search?.focus({ preventScroll: true });
		});
	});

	// Keyboard navigation follows the on-screen row order, not the flat option list order.
	let visualOrder = $derived.by(() => {
		const order: string[] = [];

		for (const item of ms.groupedFilteredOptions.loaded) order.push(item.option.id);
		for (const item of ms.groupedFilteredOptions.favorites) order.push(item.option.id);
		for (const group of ms.groupedFilteredOptions.available) {
			for (const item of group.items) order.push(item.option.id);
		}

		return order;
	});

	let highlightedIndex = $derived(highlightedId ? visualOrder.indexOf(highlightedId) : -1);

	function moveHighlight(direction: 1 | -1) {
		const len = visualOrder.length;

		if (len === 0) {
			highlightedId = null;

			return;
		}

		let index = highlightedIndex;

		if (index === -1) {
			index = direction === 1 ? 0 : len - 1;
		} else {
			index = (index + direction + len) % len;
		}

		highlightedId = visualOrder[index];
	}

	// Alt+Enter only unloads and keeps the dropdown open.
	async function handleModelKeyAction(modelId: string, unload: boolean) {
		if (!unload) {
			void ms.handleSelect(modelId);

			return;
		}

		const model = modelsStore.routerModels.find((m) => m.id === modelId);
		const status = model?.status?.value as ServerModelStatus | undefined;

		if (status === ServerModelStatus.LOADING) return;

		await modelsStore.status.unload(modelId);
	}

	export function open() {
		ms.handleOpenChange(true);
	}

	function handleSearchKeyDown(event: KeyboardEvent) {
		if (event.isComposing) return;

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			moveHighlight(1);
		} else if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			moveHighlight(-1);
		} else if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();

			if (highlightedId) {
				void handleModelKeyAction(highlightedId, event.altKey);
			} else if (visualOrder.length > 0) {
				highlightedId = visualOrder[0];
			}
		}
	}
</script>

<div class={['relative inline-flex flex-col items-end gap-1', className]}>
	{#if ms.loading && ms.options.length === 0 && ms.isRouter}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />

			Loading models…
		</div>
	{:else if ms.options.length === 0 && ms.isRouter}
		{#if currentModel}
			<span
				class={[
					'inline-flex items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground',
					className
				]}
				style="max-width: min(calc(100cqw - 10rem), 20rem)"
			>
				<MODEL_SELECTOR_ICON class="h-3.5 w-3.5 shrink-0" />
			</span>
		{:else}
			<p class="text-xs text-muted-foreground">No models available.</p>
		{/if}
	{:else}
		{@const selectedOption = ms.getDisplayOption()}
		{@const triggerModel = selectedOption?.model}
		{@const triggerStatus = triggerModel
			? modelsStore.routerModels.find((m) => m.id === triggerModel)?.status?.value
			: undefined}
		{@const triggerLoading =
			!!triggerModel &&
			(triggerStatus === ServerModelStatus.LOADING ||
				modelsStore.status.isOperationInProgress(triggerModel))}
		{@const triggerLoadPercent = triggerLoading
			? Math.round(modelLoadFraction(modelsStore.status.getLoadProgress(triggerModel)) * 100)
			: 0}

		{#if ms.isRouter}
			<DropdownMenu.Root bind:open={isOpen} onOpenChange={ms.handleOpenChange}>
				<Tooltip.Root>
					<Tooltip.Trigger>
						<!-- prevent another nested button element -->
						{#snippet child({ props })}
							<DropdownMenu.Trigger
								{...props}
								class={[
									`relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
									!ms.isCurrentModelInCache
										? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
										: forceForegroundText
											? 'text-foreground'
											: ms.isHighlightedCurrentModelActive
												? 'text-foreground'
												: 'text-foreground',
									isOpen && 'text-foreground',
									'max-w-[min(calc(100vw-4rem) md:max-w-[min(calc(100cqw-9rem),25rem)]'
								]}
								disabled={disabled || ms.updating}
							>
								<MODEL_SELECTOR_ICON class="h-3.5 w-3.5 shrink-0" />

								{#if selectedOption}
									<ModelId
										class="min-w-0 overflow-hidden"
										hideOrgName={false}
										hideQuantization
										modelId={selectedOption.model}
									/>
								{:else}
									<span class="min-w-0 font-medium">Select model</span>
								{/if}

								{#if ms.updating || ms.isLoadingModel}
									<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
								{:else}
									<ChevronDown class="h-3 w-3.5 shrink-0" />
								{/if}

								{#if triggerLoading}
									<ModelLoadHighlight percent={triggerLoadPercent} />
								{/if}
							</DropdownMenu.Trigger>
						{/snippet}
					</Tooltip.Trigger>

					{#if selectedOption}
						<Tooltip.Content>
							<p class="font-mono">{selectedOption.model}</p>
						</Tooltip.Content>
					{/if}
				</Tooltip.Root>

				<DropdownMenu.Content
					align="end"
					class="w-full max-w-[100vw] pt-0 sm:w-max sm:max-w-[calc(100vw-2rem)]"
					onOpenAutoFocus={(event) => event.preventDefault()}
				>
					<DropdownMenuSearchable
						emptyMessage="No models found."
						isEmpty={ms.filteredOptions.length === 0 && ms.isCurrentModelInCache}
						onSearchChange={(v) => ms.setSearchTerm(v)}
						onSearchKeyDown={handleSearchKeyDown}
						placeholder="Search models..."
						searchValue={ms.searchTerm}
					>
						<div class="models-list">
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
									<ModelId class="flex-1" hideQuantization modelId={currentModel} />

									<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
								</button>
							{/if}

							{#if ms.filteredOptions.length === 0}
								<p class="px-4 py-3 text-sm text-muted-foreground">No models found.</p>
							{/if}

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
									onInfoClick={ms.handleInfoClick}
									onKeyDown={(event) => {
										if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) {
											event.preventDefault();
											void handleModelKeyAction(option.id, event.altKey);
										}
									}}
									onMouseEnter={() => (highlightedId = option.id)}
									onSelect={ms.handleSelect}
									{option}
								/>
							{/snippet}

							<ModelsSelectorList
								activeId={ms.activeId}
								{currentModel}
								groups={ms.groupedFilteredOptions}
								onInfoClick={ms.handleInfoClick}
								onSelect={ms.handleSelect}
								renderOption={modelOption}
								sectionHeaderClass="my-1.5 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none"
							/>
						</div>
					</DropdownMenuSearchable>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{:else}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<!-- prevent another nested button element -->
					{#snippet child({ props })}
						<button
							{...props}
							class={[
								`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
								!ms.isCurrentModelInCache
									? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
									: forceForegroundText
										? 'text-foreground'
										: ms.isHighlightedCurrentModelActive
											? 'text-foreground'
											: 'text-foreground',
								isOpen && 'text-foreground'
							]}
							disabled={disabled || ms.updating}
							onclick={() => ms.handleOpenChange(true)}
							style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
						>
							<MODEL_SELECTOR_ICON class="h-3.5 w-3.5 shrink-0" />

							{#if selectedOption}
								<ModelId
									class="min-w-0 overflow-hidden"
									hideOrgName={false}
									hideQuantization
									modelId={selectedOption.model}
								/>
							{/if}

							{#if ms.updating}
								<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
							{/if}
						</button>
					{/snippet}
				</Tooltip.Trigger>

				{#if selectedOption}
					<Tooltip.Content>
						<p class="font-mono">{selectedOption.model}</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>
		{/if}
	{/if}
</div>

{#if ms.showModelDialog}
	<DialogModelInformation
		modelId={ms.infoModelId}
		onOpenChange={(v) => ms.setShowModelDialog(v)}
		open={ms.showModelDialog}
	/>
{/if}
