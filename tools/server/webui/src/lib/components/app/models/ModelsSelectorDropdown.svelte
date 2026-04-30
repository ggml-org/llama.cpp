<script lang="ts">
	import { ChevronDown, Loader2, Package } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { KeyboardKey } from '$lib/enums';
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';
	import {
		DialogModelInformation,
		DropdownMenuSearchable,
		ModelId,
		ModelsSelectorList,
		ModelsSelectorOption
	} from '$lib/components/app';
	import type { ModelItem } from './utils';

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
	let highlightedIndex = $state<number>(-1);

	const ms = useModelsSelector({
		currentModel: () => currentModel,
		useGlobalSelection: () => useGlobalSelection,
		onModelChange: () => onModelChange,
		onOpenChange: (open) => {
			isOpen = open;
			highlightedIndex = -1;
		}
	});

	$effect(() => {
		void ms.searchTerm;
		highlightedIndex = -1;
	});

	export function open() {
		ms.handleOpenChange(true);
	}

	function handleSearchKeyDown(event: KeyboardEvent) {
		if (event.isComposing) return;

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();

			if (ms.filteredOptions.length === 0) return;

			if (highlightedIndex === -1 || highlightedIndex === ms.filteredOptions.length - 1) {
				highlightedIndex = 0;
			} else {
				highlightedIndex += 1;
			}
		} else if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();

			if (ms.filteredOptions.length === 0) return;

			if (highlightedIndex === -1 || highlightedIndex === 0) {
				highlightedIndex = ms.filteredOptions.length - 1;
			} else {
				highlightedIndex -= 1;
			}
		} else if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();

			if (highlightedIndex >= 0 && highlightedIndex < ms.filteredOptions.length) {
				const option = ms.filteredOptions[highlightedIndex];

				ms.handleSelect(option.id);
			} else if (ms.filteredOptions.length > 0) {
				highlightedIndex = 0;
			}
		}
	}
</script>

<div class={cn('relative inline-flex flex-col items-end gap-1', className)}>
	{#if ms.loading && ms.options.length === 0 && ms.isRouter}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />

			Loading models…
		</div>
	{:else if ms.options.length === 0 && ms.isRouter}
		{#if currentModel}
			<span
				class={cn(
					'inline-flex items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground',
					className
				)}
				style="max-width: min(calc(100cqw - 10rem), 20rem)"
			>
				<Package class="h-3.5 w-3.5" />

				<ModelId modelId={currentModel} class="min-w-0" hideQuantization />
			</span>
		{:else}
			<p class="text-xs text-muted-foreground">No models available.</p>
		{/if}
	{:else}
		{@const selectedOption = ms.getDisplayOption()}

		{#if ms.isRouter}
			<DropdownMenu.Root bind:open={isOpen} onOpenChange={ms.handleOpenChange}>
				<DropdownMenu.Trigger
					class={cn(
						`inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
						!ms.isCurrentModelInCache
							? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
							: forceForegroundText
								? 'text-foreground'
								: ms.isHighlightedCurrentModelActive
									? 'text-foreground'
									: 'text-muted-foreground',
						isOpen ? 'text-foreground' : '',
						'max-w-[min(calc(100vw-4rem) md:max-w-[min(calc(100cqw-9rem),25rem)]'
					)}
					disabled={disabled || ms.updating}
				>
					<Package class="h-3.5 w-3.5" />

					{#if selectedOption}
						<Tooltip.Root>
							<Tooltip.Trigger>
								<!-- prevent another nested button element -->
								{#snippet child({ props })}
									<ModelId
										modelId={selectedOption.model}
										class="min-w-0 overflow-hidden"
										hideOrgName={false}
										{...props}
									/>
								{/snippet}
							</Tooltip.Trigger>

							<Tooltip.Content>
								<p class="font-mono">{selectedOption.model}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{:else}
						<span class="min-w-0 font-medium">Select model</span>
					{/if}

					{#if ms.updating || ms.isLoadingModel}
						<Loader2 class="h-3 w-3.5 animate-spin" />
					{:else}
						<ChevronDown class="h-3 w-3.5" />
					{/if}
				</DropdownMenu.Trigger>

				<DropdownMenu.Content
					align="end"
					class="w-full max-w-[100vw] pt-0 sm:w-max sm:max-w-[calc(100vw-2rem)]"
				>
					<DropdownMenuSearchable
						searchValue={ms.searchTerm}
						onSearchChange={(v) => ms.setSearchTerm(v)}
						placeholder="Search models..."
						onSearchKeyDown={handleSearchKeyDown}
						emptyMessage="No models found."
						isEmpty={ms.filteredOptions.length === 0 && ms.isCurrentModelInCache}
					>
						<div class="models-list">
							{#if !ms.isCurrentModelInCache && currentModel}
								<!-- Show unavailable model as first option (disabled) -->
								<button
									type="button"
									class="flex w-full cursor-not-allowed items-center bg-red-400/10 p-2 text-left text-sm text-red-400"
									role="option"
									aria-selected="true"
									aria-disabled="true"
									disabled
								>
									<ModelId modelId={currentModel} class="flex-1" hideQuantization />

									<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
								</button>
							{/if}

							{#if ms.filteredOptions.length === 0}
								<p class="px-4 py-3 text-sm text-muted-foreground">No models found.</p>
							{/if}

							{#snippet modelOption(item: ModelItem, hideOrgName: boolean)}
								{@const { option, flatIndex } = item}
								{@const isSelected = currentModel === option.model || ms.activeId === option.id}
								{@const isHighlighted = flatIndex === highlightedIndex}
								{@const isFav = ms.isFavorite(option.model)}

								<ModelsSelectorOption
									{option}
									{isSelected}
									{isHighlighted}
									{isFav}
									{hideOrgName}
									onSelect={ms.handleSelect}
									onInfoClick={ms.handleInfoClick}
									onMouseEnter={() => (highlightedIndex = flatIndex)}
									onKeyDown={(e) => {
										if (e.key === KeyboardKey.ENTER || e.key === KeyboardKey.SPACE) {
											e.preventDefault();
											ms.handleSelect(option.id);
										}
									}}
								/>
							{/snippet}

							<ModelsSelectorList
								groups={ms.groupedFilteredOptions}
								{currentModel}
								activeId={ms.activeId}
								sectionHeaderClass="my-1.5 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none"
								onSelect={ms.handleSelect}
								onInfoClick={ms.handleInfoClick}
								renderOption={modelOption}
							/>
						</div>
					</DropdownMenuSearchable>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{:else}
			<button
				class={cn(
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!ms.isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: ms.isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground',
					isOpen ? 'text-foreground' : ''
				)}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				onclick={() => ms.handleOpenChange(true)}
				disabled={disabled || ms.updating}
			>
				<Package class="h-3.5 w-3.5" />

				{#if selectedOption}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<!-- prevent another nested button element -->
							{#snippet child({ props })}
								<ModelId
									modelId={selectedOption.model}
									class="min-w-0 overflow-hidden"
									hideOrgName={false}
									{...props}
								/>
							{/snippet}
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p class="font-mono">{selectedOption.model}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if ms.updating}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{/if}
			</button>
		{/if}
	{/if}
</div>

{#if ms.showModelDialog}
	<DialogModelInformation
		open={ms.showModelDialog}
		onOpenChange={(v) => ms.setShowModelDialog(v)}
		modelId={ms.infoModelId}
	/>
{/if}
