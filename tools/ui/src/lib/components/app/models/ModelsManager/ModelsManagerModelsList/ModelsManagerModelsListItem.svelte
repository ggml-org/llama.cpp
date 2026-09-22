<script lang="ts">
	import { modelQuantLabel, modelSizeLabel } from '../utils';
	import { ChevronDown, ChevronUp, Heart } from '@lucide/svelte';
	import { ModelCapabilityIcons } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isFavorite: boolean;
		isSelected: boolean;
		onSelect: () => void;
		onToggleFavorite: () => void;
		option: ModelOption;
	}

	let { isFavorite, isSelected, onSelect, onToggleFavorite, option }: Props = $props();

	let expanded = $state(true);

	let serverStatus = $derived.by(() => {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isFailed = $derived(serverStatus === ServerModelStatus.FAILED);
	let isSleeping = $derived(serverStatus === ServerModelStatus.SLEEPING);
	let isLoaded = $derived(
		(serverStatus === ServerModelStatus.LOADED || isSleeping) && !isOperationInProgress
	);
	let isLoading = $derived(serverStatus === ServerModelStatus.LOADING || isOperationInProgress);

	let quant = $derived(modelQuantLabel(option));
	let size = $derived(modelSizeLabel(option));
	let supportsToolUse = $derived(option.capabilities.includes(ModelCapability.TOOL_USE));
	let supportsThinking = $derived(option.capabilities.includes(ModelCapability.REASONING));
</script>

<div
	class={[
		'group rounded-md transition',
		isSelected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/50'
	]}
>
	<div class="flex items-center gap-1.5 px-2 py-1.5">
		<button
			aria-label="Select model"
			class="shrink-0 cursor-pointer"
			onclick={onSelect}
			type="button"
		>
			{#if isLoading}
				<span class="block h-2 w-2 animate-pulse rounded-full bg-amber-500"></span>
			{:else if isFailed}
				<span class="block h-2 w-2 rounded-full bg-destructive"></span>
			{:else}
				<span
					class="block h-2 w-2 rounded-full {isLoaded
						? 'bg-emerald-500'
						: 'border border-muted-foreground/50'}"
				></span>
			{/if}
		</button>

		{#if isFavorite}
			<button
				aria-label="Remove from favorites"
				class="shrink-0 cursor-pointer text-muted-foreground transition hover:text-foreground"
				onclick={onToggleFavorite}
				type="button"
			>
				<Heart class="h-3.5 w-3.5 fill-current" />
			</button>
		{/if}

		<button class="min-w-0 flex-1 cursor-pointer text-left" onclick={onSelect} type="button">
			<span class="block truncate text-sm">{option.name}</span>
		</button>

		<button
			aria-label={expanded ? 'Hide model details' : 'Show model details'}
			class="shrink-0 cursor-pointer text-muted-foreground transition hover:text-foreground"
			onclick={() => (expanded = !expanded)}
			type="button"
		>
			{#if expanded}
				<ChevronUp class="h-3 w-3" />
			{:else}
				<ChevronDown class="h-3 w-3" />
			{/if}
		</button>
	</div>

	{#if expanded}
		<div class="flex flex-wrap items-center gap-1.5 px-2 pb-2 pl-5.5">
			{#if quant}
				<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">{quant}</Badge>
			{/if}

			{#if size}
				<span class="text-xs text-muted-foreground">{size}</span>
			{/if}

			<ModelCapabilityIcons modalities={option.modalities} {supportsThinking} {supportsToolUse} />
		</div>
	{/if}
</div>
