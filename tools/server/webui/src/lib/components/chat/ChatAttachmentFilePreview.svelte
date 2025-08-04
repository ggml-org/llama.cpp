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
		textContent?: string;
		onClick?: () => void;
	}

	let {
		id,
		name,
		type,
		size,
		readonly = false,
		onRemove,
		class: className = '',
		textContent,
		onClick
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

	function getPreviewText(content: string): string {
		// Get first 150 characters for preview
		return content.length > 150 ? content.substring(0, 150) : content;
	}
</script>

{#if type === 'text/plain' || type === 'text'}
	{#if readonly}
		<button
			class="bg-muted border-border cursor-pointer rounded-lg border p-3 transition-shadow hover:shadow-md {className} w-full max-w-2xl"
			onclick={onClick}
			aria-label={`Preview ${name}`}
			type="button"
		>
			<div class="flex items-start gap-3">
				<div class="flex min-w-0 flex-1 flex-col items-start text-left">
					<span class="text-foreground w-full truncate text-sm font-medium">{name}</span>
					{#if size}
						<span class="text-muted-foreground text-xs">{formatFileSize(size)}</span>
					{/if}
					{#if textContent && type === 'text'}
						<div class="relative mt-2 w-full">
							<div
								class="text-muted-foreground overflow-hidden whitespace-pre-wrap break-words font-mono text-xs leading-relaxed"
							>
								{getPreviewText(textContent)}
							</div>
							{#if textContent.length > 150}
								<div
									class="from-muted pointer-events-none absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t to-transparent"
								></div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
		</button>
	{:else}
		<!-- Non-readonly mode (ChatForm) -->
		<div class="bg-muted border-border relative rounded-lg border p-3 {className} w-64">
			<!-- Remove button in top-right corner -->
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="absolute right-2 top-2 h-6 w-6 bg-white/20 p-0 hover:bg-white/30"
				onclick={() => onRemove?.(id)}
				aria-label="Remove file"
			>
				<X class="h-3 w-3" />
			</Button>

			<!-- Content -->
			<div class="pr-8">
				<!-- Add right padding to avoid overlap with X button -->
				<span class="text-foreground mb-3 block truncate text-sm font-medium">{name}</span>

				{#if textContent}
					<div class="relative">
						<div
							class="text-muted-foreground overflow-hidden whitespace-pre-wrap break-words font-mono text-xs leading-relaxed"
							style="max-height: 3.6em; line-height: 1.2em;"
						>
							{getPreviewText(textContent)}
						</div>
						{#if textContent.length > 150}
							<div
								class="from-muted pointer-events-none absolute bottom-0 left-0 right-0 h-4 bg-gradient-to-t to-transparent"
							></div>
						{/if}
					</div>
				{/if}
			</div>
		</div>
	{/if}
{:else}
	<div class="bg-muted border-border flex items-center gap-2 rounded-lg border p-2 {className}">
		<div
			class="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded text-xs font-medium"
		>
			{getFileTypeLabel(type)}
		</div>
		<div class="flex flex-col">
			<span class="text-foreground max-w-72 truncate text-sm font-medium">{name}</span>
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
{/if}
