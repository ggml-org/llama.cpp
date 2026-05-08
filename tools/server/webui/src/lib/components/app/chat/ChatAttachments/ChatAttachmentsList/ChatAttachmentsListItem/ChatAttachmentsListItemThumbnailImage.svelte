<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import { X } from '@lucide/svelte';

	interface Props {
		class?: string;
		height?: string;
		id: string;
		imageClass?: string;
		onclick?: (event?: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		preview: string;
		readonly?: boolean;
		width?: string;
	}

	let {
		class: className = '',
		height = 'h-16',
		id,
		imageClass = '',
		onclick,
		onRemove,
		name,
		preview,
		readonly = false,
		width = 'w-auto'
	}: Props = $props();
</script>

{#snippet image()}
	<img src={preview} alt={name} class="{height} {width} cursor-pointer object-cover {imageClass}" />
{/snippet}

<div
	class="group relative overflow-hidden rounded-lg bg-muted shadow-lg dark:border dark:border-muted {className}"
>
	{#if onclick}
		<button
			aria-label="Preview {name}"
			class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			{onclick}
			type="button"
		>
			{@render image()}
		</button>
	{:else}
		{@render image()}
	{/if}

	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<ActionIcon
				class="text-white"
				icon={X}
				onclick={() => onRemove?.(id)}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>
