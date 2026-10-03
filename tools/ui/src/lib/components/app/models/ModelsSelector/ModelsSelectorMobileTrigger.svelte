<script lang="ts">
	import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import { ModelId, ModelsSelectorTriggerIcon } from '$lib/components/app';
	import { ServerModelStatus } from '$lib/enums';
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';
	import { modelsStore, uiStore } from '$lib/stores';
	import { modelLoadFraction } from '$lib/utils';

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
		disabled = false,
		forceForegroundText = false,
		onModelChange,
		useGlobalSelection = false
	}: Props = $props();

	const ms = useModelsSelector({
		currentModel: () => currentModel,
		onModelChange: () => onModelChange,
		useGlobalSelection: () => useGlobalSelection
	});

	const selectedOption = $derived(ms.getDisplayOption());

	/**
	 * A phone picks its model in the manager: the list, the search and the row actions
	 * already live there, so the trigger opens it. It opens on the table, not on a
	 * model: the pane is for the model the user picks there.
	 */
	function openManager() {
		uiStore.openModelsManager();
	}

	export function open() {
		openManager();
	}
</script>

<div class={['relative inline-flex flex-col items-end gap-1', className]}>
	{#if ms.loading && ms.options.length === 0 && ms.isMultiModel}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />

			Loading models...
		</div>
	{:else if ms.options.length === 0 && ms.isMultiModel}
		<span class="text-xs text-muted-foreground">No models yet.</span>
	{:else}
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

		{#if ms.isMultiModel}
			<button
				class={[
					`relative inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 max-sm:px-3 max-sm:py-2 max-sm:text-sm dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
					!ms.isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: ms.isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-foreground'
				]}
				disabled={disabled || ms.updating}
				onclick={openManager}
				style="max-width: min(calc(100cqw - 9rem), 20rem)"
				type="button"
			>
				<ModelsSelectorTriggerIcon class="h-3.5 w-3.5 shrink-0" option={selectedOption} />

				{#if !selectedOption}
					<span class="min-w-0 font-medium">Select model</span>
				{:else}
					<ModelId
						class="text-xs"
						hideOrgName
						hideQuantization
						hideTags
						modelId={selectedOption.model}
					/>
				{/if}

				{#if ms.updating || ms.isLoadingModel}
					<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
				{:else}
					<ChevronDown class="h-3 w-3.5 shrink-0" />
				{/if}

				{#if triggerLoading}
					<ModelLoadHighlight percent={triggerLoadPercent} />
				{/if}
			</button>
		{:else}
			<!-- a single-model server has no list: the trigger opens the manager instead -->
			<button
				class={[
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
					!ms.isCurrentModelInCache
						? 'bg-red-400/10 text-red-400! hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: ms.isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-foreground'
				]}
				disabled={disabled || ms.updating}
				onclick={() => ms.handleOpenChange(true)}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				type="button"
			>
				<ModelsSelectorTriggerIcon class="h-3.5 w-3.5 shrink-0" option={selectedOption} />

				<ModelId
					class="font-medium"
					hideOrgName
					hideQuantization
					hideTags
					modelId={selectedOption?.model || ''}
				/>

				{#if ms.updating}
					<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
				{/if}
			</button>
		{/if}
	{/if}
</div>
