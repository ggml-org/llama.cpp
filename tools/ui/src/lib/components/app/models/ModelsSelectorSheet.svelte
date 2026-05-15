<script lang="ts">
	import { ChevronDown, Loader2, Package } from '@lucide/svelte';
	import * as Sheet from '$lib/components/ui/sheet';
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';
	import {
		DialogModelInformation,
		ModelId,
		ModelsSelectorList,
		SearchInput,
		TruncatedText
	} from '$lib/components/app';

	interface Props {
		class?: string;
		currentModel?: string | null;
		/** Callback when model changes. Return false to keep menu open (e.g., for validation failures) */
		onModelChange?: (modelId: string, modelName: string) => Promise<boolean> | boolean | void;
		disabled?: boolean;
		forceForegroundText?: boolean;
		/** When true, user's global selection takes priority over currentModel (for form selector) */
		useGlobalSelection?: boolean;
	}

	let {
		class: className = '',
		currentModel = null,
		onModelChange,
		disabled = false,
		forceForegroundText = false,
		useGlobalSelection = false
	}: Props = $props();

	let sheetOpen = $state(false);

	const ms = useModelsSelector({
		currentModel: () => currentModel,
		useGlobalSelection: () => useGlobalSelection,
		onModelChange: () => onModelChange,
		onOpenChange: (open) => {
			sheetOpen = open;
		}
	});

	export function open() {
		ms.handleOpenChange(true);
	}

	function handleSheetOpenChange(open: boolean) {
		if (!open) {
			ms.handleOpenChange(false);
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
		<p class="text-xs text-muted-foreground">No models available.</p>
	{:else}
		{@const selectedOption = ms.getDisplayOption()}

		{#if ms.isRouter}
			<button
				type="button"
				class={[
					`inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
					!ms.isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: ms.isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-foreground',
					sheetOpen && 'text-foreground'
				]}
				style="max-width: min(calc(100cqw - 9rem), 20rem)"
				disabled={disabled || ms.updating}
				onclick={() => ms.handleOpenChange(true)}
			>
				<Package class="h-3.5 w-3.5" />

				{#if !selectedOption}
					<span class="min-w-0 font-medium">Select model</span>
				{:else}
					<ModelId
						class="text-xs"
						modelId={selectedOption?.model || ''}
						hideQuantization
						hideOrgName
					/>
				{/if}

				{#if ms.updating || ms.isLoadingModel}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{:else}
					<ChevronDown class="h-3 w-3.5" />
				{/if}
			</button>

			<Sheet.Root bind:open={sheetOpen} onOpenChange={handleSheetOpenChange}>
				<Sheet.Content side="bottom" class="max-h-[85vh] gap-1">
					<Sheet.Header>
						<Sheet.Title>Select Model</Sheet.Title>

						<Sheet.Description class="sr-only">
							Choose a model to use for the conversation
						</Sheet.Description>
					</Sheet.Header>

					<div class="flex flex-col gap-1 pb-4">
						<div class="mb-3 px-4">
							<SearchInput
								placeholder="Search models..."
								value={ms.searchTerm}
								onInput={(v) => ms.setSearchTerm(v)}
							/>
						</div>

						<div class="max-h-[60vh] overflow-y-auto px-2">
							{#if !ms.isCurrentModelInCache && currentModel}
								<button
									type="button"
									class="flex w-full cursor-not-allowed items-center rounded-md bg-red-400/10 px-3 py-2.5 text-left text-sm text-red-400"
									disabled
								>
									<span class="min-w-0 flex-1 truncate">
										{selectedOption?.name || currentModel}
									</span>
									<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
								</button>
								<div class="my-1 h-px bg-border"></div>
							{/if}

							{#if ms.filteredOptions.length === 0}
								<p class="px-3 py-3 text-center text-sm text-muted-foreground">No models found.</p>
							{/if}

							<ModelsSelectorList
								groups={ms.groupedFilteredOptions}
								{currentModel}
								activeId={ms.activeId}
								sectionHeaderClass="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none"
								orgHeaderClass="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none [&:not(:first-child)]:mt-2"
								onSelect={ms.handleSelect}
								onInfoClick={ms.handleInfoClick}
							/>
						</div>
					</div>
				</Sheet.Content>
			</Sheet.Root>
		{:else}
			<button
				class={[
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
					!ms.isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: ms.isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-foreground'
				]}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				onclick={() => ms.handleOpenChange(true)}
				disabled={disabled || ms.updating}
			>
				<Package class="h-3.5 w-3.5" />

				<TruncatedText text={selectedOption?.model || ''} class="font-medium" />

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
