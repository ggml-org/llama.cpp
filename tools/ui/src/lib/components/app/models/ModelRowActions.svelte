<script lang="ts">
	import { Heart, HeartOff, Info } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isFav: boolean;
		isLoaded: boolean;
		/** Renders the info action for a loaded model, when the caller can show it. */
		onInfo?: (model: string) => void;
		option: ModelOption;
		/** Selector rows reveal the actions on hover, table rows keep them visible. */
		revealOnHover?: boolean;
	}

	let { isFav, isLoaded, onInfo, option, revealOnHover = true }: Props = $props();
</script>

<div
	class={[
		'flex items-center justify-center gap-1',
		revealOnHover
			? 'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100'
			: ''
	]}
	onclick={(event) => event.stopPropagation()}
	onkeydown={(event) => event.stopPropagation()}
	role="presentation"
>
	{#if isLoaded && onInfo}
		<ActionIcon
			class="h-5 w-5 hover:text-foreground"
			icon={Info}
			iconSize="h-4 w-4"
			onclick={() => onInfo?.(option.model)}
			tooltip="Model information"
			tooltipAsTitle
		/>
	{/if}

	{#if isFav}
		<span class="flex h-5 w-5 items-center justify-center">
			<span class="flex group-hover:hidden [@media(pointer:coarse)]:hidden">
				<ActionIcon
					class="h-5 w-5 text-rose-500 hover:text-foreground"
					icon={Heart}
					iconSize="h-4 w-4"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Remove from favorites"
					tooltipAsTitle
				/>
			</span>

			<span class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
				<ActionIcon
					class="h-5 w-5 hover:text-foreground"
					icon={HeartOff}
					iconSize="h-4 w-4"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Remove from favorites"
					tooltipAsTitle
				/>
			</span>
		</span>
	{:else}
		<ActionIcon
			class="h-5 w-5 hover:text-foreground"
			icon={Heart}
			iconSize="h-4 w-4"
			onclick={() => modelsStore.toggleFavorite(option.model)}
			tooltip="Add to favorites"
			tooltipAsTitle
		/>
	{/if}
</div>
