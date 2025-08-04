<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { X } from '@lucide/svelte';

	interface Props {
		id: string;
		name: string;
		preview: string;
		size?: number;
		readonly?: boolean;
		onRemove?: (id: string) => void;
		onClick?: () => void;
		class?: string;
		// Customizable size props
		width?: string;
		height?: string;
		imageClass?: string;
	}

	let {
		id,
		name,
		preview,
		size,
		readonly = false,
		onRemove,
		onClick,
		class: className = '',
		// Default to small size for form previews
		width = 'w-auto',
		height = 'h-24',
		imageClass = ''
	}: Props = $props();
</script>

<div class="bg-muted border-border relative overflow-hidden rounded-lg border {className}">
	{#if onClick}
		<button
			type="button"
			class="focus:ring-primary block h-full w-full rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2"
			onclick={onClick}
			aria-label="Preview {name}"
		>
			<img
				src={preview}
				alt={name}
				class="{height} {width} cursor-pointer object-cover {imageClass}"
			/>
		</button>
	{:else}
		<img
			src={preview}
			alt={name}
			class="{height} {width} cursor-pointer object-cover {imageClass}"
		/>
	{/if}
	{#if !readonly}
		<div
			class="absolute right-1 top-1 flex items-center justify-center opacity-0 transition-opacity hover:opacity-100"
		>
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="h-6 w-6 bg-white/20 p-0 text-white hover:bg-white/30"
				onclick={() => onRemove?.(id)}
			>
				<X class="h-3 w-3" />
			</Button>
		</div>
	{/if}
</div>
