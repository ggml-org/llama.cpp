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

<div class="bg-muted border-border relative overflow-hidden rounded-lg border {className}">
	<img src={preview} alt={name} class="{height} {width} object-cover {imageClass}" />
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
