<script lang="ts">
	import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import {
		ModelId,
		ModelsSelectorDrawer,
		ModelsSelectorDropdown,
		ModelsSelectorList,
		ModelsSelectorOption,
		ModelsSelectorTriggerIcon
	} from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { DROPDOWN_MENU_CONTENT_SEARCH_SELECTOR, MODEL_ICON, SETTINGS_KEYS } from '$lib/constants';
	import { KeyboardKey, ServerModelStatus } from '$lib/enums';
	import { useChatFormModel } from '$lib/hooks/use-chat-form-model.svelte';
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';
	import { deviceStore, modelsStore, serverStore, settingsStore } from '$lib/stores';
	import { type ModelItem, modelLoadFraction } from '$lib/utils';

	interface Props {
		/** Model to show, when the caller owns it, e.g. the model of one message. */
		currentModel?: string | null;
		disabled?: boolean;
		forceForegroundText?: boolean;
		onModelChange?: (modelId: string, modelName: string) => Promise<boolean> | boolean | void;
		useGlobalSelection?: boolean;
	}

	let {
		currentModel = null,
		disabled = false,
		forceForegroundText = false,
		onModelChange,
		useGlobalSelection = false
	}: Props = $props();

	// one setting for every model id in the selector: the trigger and the rows
	let showOrgName = $derived(settingsStore.config[SETTINGS_KEYS.SHOW_MODEL_ORG_NAME] ?? true);

	// a phone opens the picker in a drawer, a desktop keeps the anchored dropdown
	let isMobile = $derived(deviceStore.isMobile);
	let isOffline = $derived(!!serverStore.error);

	let isOpen = $state(false);
	let highlightedId = $state<string | null>(null);

	const formModel = useChatFormModel();

	// the store's pick wins while it differs from the conversation's model: the
	// trigger follows the model the user just picked, whose load state and
	// progress it reports, not the one the last answer came from
	let selectorModel = $derived.by(() => {
		const storeModel = modelsStore.selectedModelName;

		if (storeModel && storeModel !== formModel.conversationModel) {
			return storeModel;
		}

		return formModel.conversationModel;
	});

	// the caller's model comes first, then the pick, then the conversation's
	let displayModel = $derived(currentModel ?? selectorModel);

	const ms = useModelsSelector({
		currentModel: () => displayModel,
		onModelChange: () => onModelChange,
		onOpenChange: (open) => {
			isOpen = open;
			highlightedId = null;
		},
		useGlobalSelection: () => useGlobalSelection
	});

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

	$effect(() => {
		void ms.searchTerm;
		highlightedId = null;
	});

	// Keyboard navigation follows the on-screen row order, not the flat option list order.
	let visualOrder = $derived.by(() => {
		const order: string[] = [];

		for (const item of ms.favoriteItems) order.push(item.option.id);
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

	// Alt+Enter only unloads and keeps the picker open.
	async function handleModelKeyAction(modelId: string, unload: boolean) {
		if (!unload) {
			void ms.handleSelect(modelId);

			return;
		}

		const status = modelsStore.getModelStatus(modelId);

		if (status === ServerModelStatus.LOADING) return;

		await modelsStore.status.unload(modelId);
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

	// bits-ui auto-focuses the opened content, which can yank the page scroll: the
	// content prevents that and this focuses the search input instead
	$effect(() => {
		if (!isOpen || isMobile) return;

		let frames = 0;
		let handle = requestAnimationFrame(function focusSearch() {
			const input = document.querySelector<HTMLElement>(DROPDOWN_MENU_CONTENT_SEARCH_SELECTOR);

			if (input) {
				input.focus({ preventScroll: true });

				return;
			}

			if (frames++ < 20) handle = requestAnimationFrame(focusSearch);
		});

		return () => cancelAnimationFrame(handle);
	});

	export function open() {
		ms.handleOpenChange(true);
	}
</script>

{#snippet selectorTriggerInner()}
	<ModelsSelectorTriggerIcon
		class="mr-0.375 size-3.5 max-md:size-4 shrink-0"
		option={selectedOption}
	/>

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
	{@const isSelected = ms.activeId === option.id}
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
				void handleModelKeyAction(option.id, event.altKey);
			}
		}}
		onMouseEnter={() => (highlightedId = option.id)}
		onSelect={ms.handleSelect}
		{option}
	/>
{/snippet}

{#snippet pickerList(listClass: string)}
	<div class={['models-list', listClass]}>
		{#if !ms.isCurrentModelInCache && displayModel}
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
					modelId={displayModel}
				/>

				<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
			</button>
		{/if}

		{#if ms.isEmpty}
			<p class="px-4 py-3 text-sm text-muted-foreground">{ms.emptyMessage}</p>
		{/if}

		<ModelsSelectorList
			activeId={ms.activeId}
			currentModel={displayModel}
			favorites={ms.favoriteItems}
			groups={ms.groupedFilteredOptions}
			loaded={ms.loadedItems}
			onSelect={ms.handleSelect}
			renderOption={modelOption}
			sectionHeaderClass="[&:not(:first-child)]:mt-2 mb-1 px-2 py-2.5 text-sm font-semibold text-foreground/80 select-none"
			{showOrgName}
		/>
	</div>
{/snippet}

{#if ms.loading && ms.options.length === 0 && ms.isMultiModel}
	<div class="flex items-center gap-2 text-xs text-muted-foreground">
		<Loader2 class="h-3.5 w-3.5 animate-spin" />

		Loading models...
	</div>
{:else if ms.options.length === 0 && ms.isMultiModel}
	{#if displayModel}
		<span
			class="inline-flex items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground"
			style="max-width: min(calc(100cqw - 10rem), 48rem)"
		>
			<MODEL_ICON class="h-3.5 w-3.5 shrink-0" />
		</span>
	{:else}
		<span class="text-xs text-muted-foreground">No models yet.</span>
	{/if}
{:else if ms.isMultiModel}
	{#if isMobile}
		<button
			class="relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:gap-1.5 max-md:h-7 max-md:px-2.25 max-md:py-1 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground"
			disabled={disabled || ms.updating}
			onclick={() => ms.handleOpenChange(true)}
			style="max-width: min(calc(100vw-4rem), 32rem)"
			type="button"
		>
			{@render selectorTriggerInner()}
		</button>

		<ModelsSelectorDrawer
			bind:open={isOpen}
			emptyMessage={ms.emptyMessage}
			isEmpty={ms.isEmpty && ms.isCurrentModelInCache}
			onOpenChange={ms.handleOpenChange}
			onSearchChange={(v) => ms.setSearchTerm(v)}
			onSearchKeyDown={handleSearchKeyDown}
			searchTerm={ms.searchTerm}
		>
			{@render pickerList('px-2.5')}
		</ModelsSelectorDrawer>
	{:else}
		<ModelsSelectorDropdown
			bind:open={isOpen}
			currentModel={displayModel}
			disabled={disabled || isOffline}
			{highlightedId}
			{ms}
			onHighlight={(id) => (highlightedId = id)}
			onModelKeyAction={(id, unload) => void handleModelKeyAction(id, unload)}
			onSearchKeyDown={handleSearchKeyDown}
			{showOrgName}
		/>
	{/if}
{:else if isMobile}
	<!-- a phone has no manager: the single model still gets the drawer, so its
	     load control, actions and information stay reachable -->
	<button
		class="relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:gap-1.5 max-md:h-7 max-md:px-2.25 max-md:py-1 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground"
		disabled={disabled || ms.updating}
		onclick={() => ms.handleOpenChange(true)}
		style="max-width: min(calc(100vw-4rem), 32rem)"
		type="button"
	>
		{@render selectorTriggerInner()}
	</button>

	<ModelsSelectorDrawer
		bind:open={isOpen}
		emptyMessage={ms.emptyMessage}
		isEmpty={ms.isEmpty && ms.isCurrentModelInCache}
		onOpenChange={ms.handleOpenChange}
		onSearchChange={(v) => ms.setSearchTerm(v)}
		onSearchKeyDown={handleSearchKeyDown}
		searchTerm={ms.searchTerm}
	>
		{@render pickerList('px-2.5')}
	</ModelsSelectorDrawer>
{:else}
	<!-- a lone llama.cpp server has nothing to choose from: the trigger opens the
	     manager instead, so the button is its own affordance -->
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				<button
					{...props}
					class={[
						`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
						!ms.isCurrentModelInCache
							? 'bg-red-400/10 text-red-400! hover:bg-red-400/20 hover:text-red-400'
							: forceForegroundText
								? 'text-foreground'
								: 'text-foreground',
						isOpen && 'text-foreground'
					]}
					disabled={disabled || ms.updating}
					onclick={() => ms.handleOpenChange(true)}
					style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				>
					<ModelsSelectorTriggerIcon class="mr-0.375 size-3.5 shrink-0" option={selectedOption} />

					{#if selectedOption}
						<ModelId
							class="min-w-0 overflow-hidden"
							hideOrgName={!showOrgName}
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
	</Tooltip.Root>
{/if}
