<script lang="ts">
	import {
		CircleAlert,
		Heart,
		HeartOff,
		Info,
		Loader2,
		Power,
		PowerOff,
		RotateCw
	} from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { ActionIcon, ModelId } from '$lib/components/app';
	import type { ModelOption } from '$lib/types/models';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, routerModels } from '$lib/stores/models.svelte';

	interface Props {
		option: ModelOption;
		isSelected: boolean;
		isHighlighted: boolean;
		isFav: boolean;
		showOrgName?: boolean;
		onSelect: (modelId: string) => void;
		onMouseEnter: () => void;
		onKeyDown: (e: KeyboardEvent) => void;
		onInfoClick?: (modelName: string) => void;
	}

	let {
		option,
		isSelected,
		isHighlighted,
		isFav,
		showOrgName = false,
		onSelect,
		onMouseEnter,
		onKeyDown,
		onInfoClick
	}: Props = $props();

	let currentRouterModels = $derived(routerModels());
	let serverStatus = $derived.by(() => {
		const model = currentRouterModels.find((m) => m.id === option.model);
		return (model?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.isModelOperationInProgress(option.model));
	let isFailed = $derived(serverStatus === ServerModelStatus.FAILED);
	let isSleeping = $derived(serverStatus === ServerModelStatus.SLEEPING);
	let isLoaded = $derived(
		(serverStatus === ServerModelStatus.LOADED || isSleeping) && !isOperationInProgress
	);
	let isLoading = $derived(serverStatus === ServerModelStatus.LOADING || isOperationInProgress);
</script>

<div
	class={cn(
		'group flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
		'cursor-pointer hover:bg-muted focus:bg-muted',
		isSelected || isHighlighted
			? 'bg-accent text-accent-foreground'
			: 'hover:bg-accent hover:text-accent-foreground',
		isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
	)}
	role="option"
	aria-selected={isSelected || isHighlighted}
	tabindex="0"
	onclick={() => onSelect(option.id)}
	onmouseenter={onMouseEnter}
	onkeydown={onKeyDown}
>
	<ModelId
		modelId={option.model}
		{showOrgName}
		aliases={option.aliases}
		tags={option.tags}
		class="flex-1"
	/>

	<div class="flex shrink-0 items-center gap-1">
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="pointer-events-none flex items-center justify-center gap-0.75 pl-2 opacity-0 group-hover:pointer-events-auto group-hover:opacity-100"
			onclick={(e) => e.stopPropagation()}
		>
			{#if isFav}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={HeartOff}
					tooltip="Remove from favorites"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => modelsStore.toggleFavorite(option.model)}
				/>
			{:else}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={Heart}
					tooltip="Add to favorites"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => modelsStore.toggleFavorite(option.model)}
				/>
			{/if}

			<!-- info button: only shown when model is loaded and callback is provided -->
			{#if isLoaded && onInfoClick}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={Info}
					tooltip="Model information"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => onInfoClick(option.model)}
				/>
			{/if}
		</div>

		{#if isLoading}
			<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
		{:else if isFailed}
			<div class="flex w-4 items-center justify-center">
				<CircleAlert class="h-3.5 w-3.5 text-red-500 group-hover:hidden" />

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="hidden group-hover:flex" onclick={(e) => e.stopPropagation()}>
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={RotateCw}
						tooltip="Retry loading model"
						class="h-3 w-3 text-red-500 hover:text-foreground"
						onclick={() => modelsStore.loadModel(option.model)}
					/>
				</div>
			</div>
		{:else if isSleeping}
			<div class="flex w-4 items-center justify-center">
				<span class="h-2 w-2 rounded-full bg-orange-400 group-hover:hidden"></span>

				<div class="hidden group-hover:flex">
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={PowerOff}
						tooltip="Unload model"
						class="h-3 w-3 text-red-500 hover:text-red-600"
						onclick={(e) => {
							e?.stopPropagation();
							modelsStore.unloadModel(option.model);
						}}
					/>
				</div>
			</div>
		{:else if isLoaded}
			<div class="flex w-4 items-center justify-center">
				<span class="h-2 w-2 rounded-full bg-green-500 group-hover:hidden"></span>

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="hidden group-hover:flex" onclick={(e) => e.stopPropagation()}>
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={PowerOff}
						tooltip="Unload model"
						class="h-3 w-3 text-red-500 hover:text-red-600"
						onclick={() => modelsStore.unloadModel(option.model)}
					/>
				</div>
			</div>
		{:else}
			<div class="flex w-4 items-center justify-center">
				<span class="h-2 w-2 rounded-full bg-muted-foreground/50 group-hover:hidden"></span>

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="hidden group-hover:flex" onclick={(e) => e.stopPropagation()}>
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={Power}
						tooltip="Load model"
						class="h-3 w-3"
						onclick={() => modelsStore.loadModel(option.model)}
					/>
				</div>
			</div>
		{/if}
	</div>
</div>
