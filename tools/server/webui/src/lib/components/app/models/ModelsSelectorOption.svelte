<script lang="ts">
	import { Heart, Loader2, Power, PowerOff } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { Badge } from '$lib/components/ui/badge';
	import { ActionIcon, ModelId } from '$lib/components/app';
	import type { ModelOption } from '$lib/types/models';
	import { modelsStore } from '$lib/stores/models.svelte';

	interface Props {
		option: ModelOption;
		isLoaded: boolean;
		isLoading: boolean;
		isSelected: boolean;
		isHighlighted: boolean;
		isFav: boolean;
		highlightedIndex: number;
		showOrgName?: boolean;
		onSelect: (modelId: string) => void;
		onMouseEnter: () => void;
		onKeyDown: (e: KeyboardEvent) => void;
	}

	let {
		option,
		isLoaded,
		isLoading,
		isSelected,
		isHighlighted,
		isFav,
		showOrgName = false,
		onSelect,
		onMouseEnter,
		onKeyDown
	}: Props = $props();
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
	<ModelId modelId={option.model} {showOrgName} class="flex-1" />

	{#if option.aliases && option.aliases.length > 0}
		<div class="flex items-center gap-1">
			{#each option.aliases as alias}
				<Badge variant="tertiary" class="px-1 py-0 font-mono text-[10px]">
					{alias}
				</Badge>
			{/each}
		</div>
	{/if}

	{#if option.tags && option.tags.length > 0}
		<div class="flex items-center gap-1">
			{#each option.tags as tag}
				<Badge variant="outline" class="px-1 py-0 font-mono text-[10px]">
					{tag}
				</Badge>
			{/each}
		</div>
	{/if}

	<div class="flex shrink-0 items-center gap-2.5">
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class={cn('flex items-center justify-center w-4 pl-2', isFav ? 'flex' : 'opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto')}
			onclick={(e) => e.stopPropagation()}
		>
			<ActionIcon
				icon={Heart}
				tooltip={isFav ? 'Remove from favourites' : 'Add to favourites'}
				class={cn('h-2 w-2', isFav ? 'fill-red-400 text-red-400 hover:fill-red-500 hover:text-red-500' : 'hover:fill-current hover:text-foreground')}
				onclick={() => modelsStore.toggleFavourite(option.model)}
			/>
		</div>
		{#if isLoading}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
				</Tooltip.Trigger>
				<Tooltip.Content class="z-[9999]">
					<p>Loading model...</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{:else if isLoaded}
			<div class="flex items-center justify-center w-4">
				<span class="h-2 w-2 rounded-full bg-green-500 group-hover:hidden"></span>

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="hidden group-hover:flex" onclick={(e) => e.stopPropagation()}>
					<ActionIcon
						icon={PowerOff}
						tooltip="Unload model"
						class="text-red-500 hover:text-red-600 h-3 w-3"
						onclick={() => modelsStore.unloadModel(option.model)}
					/>
				</div>
			</div>
		{:else}
			<div class="flex items-center justify-center w-4">
				<span class="h-2 w-2 rounded-full bg-muted-foreground/50 group-hover:hidden"></span>

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="hidden group-hover:flex" onclick={(e) => e.stopPropagation()}>
					<ActionIcon
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