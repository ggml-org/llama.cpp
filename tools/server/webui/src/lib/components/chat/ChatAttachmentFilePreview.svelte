<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { X } from '@lucide/svelte';

	interface Props {
		id: string;
		name: string;
		type: string;
		size?: number;
		readonly?: boolean;
		onRemove?: (id: string) => void;
		class?: string;
	}

	let {
		id,
		name,
		type,
		size,
		readonly = false,
		onRemove,
		class: className = ''
	}: Props = $props();

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

	function getFileTypeLabel(fileType: string): string {
		return fileType.split('/').pop()?.toUpperCase() || 'FILE';
	}
</script>

<div class="bg-muted border-border flex items-center gap-2 rounded-lg border p-2 {className}">
	<div class="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded text-xs font-medium">
		{getFileTypeLabel(type)}
	</div>
	<div class="flex flex-col">
		<span class="text-foreground text-sm font-medium truncate max-w-48">{name}</span>
		{#if size}
			<span class="text-muted-foreground text-xs">{formatFileSize(size)}</span>
		{/if}
	</div>
	{#if !readonly}
		<Button
			type="button"
			variant="ghost"
			size="sm"
			class="h-6 w-6 p-0"
			onclick={() => onRemove?.(id)}
		>
			<X class="h-3 w-3" />
		</Button>
	{/if}
</div>
