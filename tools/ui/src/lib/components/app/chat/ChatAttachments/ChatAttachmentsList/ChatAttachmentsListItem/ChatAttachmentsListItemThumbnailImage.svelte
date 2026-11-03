<script lang="ts">
	import { X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';

	interface Props {
		class?: string;
		height?: string;
		id: string;
		imageClass?: string;
		label?: string;
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
		label,
		name,
		onclick,
		onRemove,
		preview,
		readonly = false,
		width = 'w-auto'
	}: Props = $props();
</script>

{#snippet image()}
	<img alt={name} class="{height} {width} cursor-pointer object-cover {imageClass}" src={preview} />
{/snippet}

<div class="group relative min-w-0 overflow-hidden rounded-lg bg-muted elevated-raised {className}">
	{#if onclick}
		<button
			aria-label="Preview {name}"
			class="block size-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			{onclick}
			type="button"
		>
			{@render image()}
		</button>
	{:else}
		{@render image()}
	{/if}

	{#if label}
		<span
			class="pointer-events-none absolute bottom-1 left-1 rounded-md bg-black/40 px-1.5 py-0.5 text-[10px] font-medium text-white matte-chip backdrop-blur-md [--badge-highlight:rgb(255_255_255/0.18)]"
		>
			{label}
		</span>
	{/if}

	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
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
