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
		class: className = '',
		// Default to small size for form previews
		width = 'w-auto',
		height = 'h-24',
		imageClass = ''
	}: Props = $props();

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
</script>

<div class="bg-muted border-border relative rounded-lg border overflow-hidden {className}">
	<img 
		src={preview} 
		alt={name} 
		class="{height} {width} object-cover {imageClass}" 
	/>
	{#if !readonly}
		<div class="absolute top-1 right-1 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="h-6 w-6 p-0 bg-white/20 hover:bg-white/30 text-white"
				onclick={() => onRemove?.(id)}
			>
				<X class="h-3 w-3" />
			</Button>
		</div>
	{/if}
	<div class="absolute bottom-0 left-0 right-0 bg-black/60 text-white p-1">
		{#if size}
			<p class="text-xs opacity-80">{formatFileSize(size)}</p>
		{/if}
	</div>
</div>
