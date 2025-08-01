<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { X } from '@lucide/svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';
	import type { DatabaseMessageExtra } from '$lib/types/database.d.ts';

	interface Props {
		// For ChatForm - pending uploads
		uploadedFiles?: ChatUploadedFile[];
		onFileRemove?: (fileId: string) => void;
		// For ChatMessage - stored attachments
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		class?: string;
	}

	let {
		uploadedFiles = [],
		onFileRemove,
		attachments = [],
		readonly = false,
		class: className = ''
	}: Props = $props();

	let displayItems = $derived(getDisplayItems());

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

	function getDisplayItems() {
		const items: Array<{
			id: string;
			name: string;
			size?: number;
			preview?: string;
			type: string;
			isImage: boolean;
		}> = [];

		// Add uploaded files (ChatForm)
		for (const file of uploadedFiles) {
			items.push({
				id: file.id,
				name: file.name,
				size: file.size,
				preview: file.preview,
				type: file.type,
				isImage: file.type.startsWith('image/')
			});
		}

		// Add stored attachments (ChatMessage)
		for (const [index, attachment] of attachments.entries()) {
			if (attachment.type === 'imageFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					preview: attachment.base64Url,
					type: 'image',
					isImage: true
				});
			} else if (attachment.type === 'textFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: 'text',
					isImage: false
				});
			} else if (attachment.type === 'audioFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: attachment.mimeType || 'audio',
					isImage: false
				});
			}
		}

		return items;
	}
</script>

{#if displayItems.length > 0}
	<div class="flex flex-wrap items-start gap-3 {className}">
		{#each displayItems as item (item.id)}
			{#if item.isImage && item.preview}
				<div class="bg-muted border-border relative rounded-lg border overflow-hidden">
					<img 
						src={item.preview} 
						alt={item.name} 
						class="h-24 w-24 object-cover" 
					/>
					{#if !readonly}
						<div class="absolute top-1 right-1 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
							<Button
								type="button"
								variant="ghost"
								size="sm"
								class="h-6 w-6 p-0 bg-white/20 hover:bg-white/30 text-white"
								onclick={() => onFileRemove?.(item.id)}
							>
								<X class="h-3 w-3" />
							</Button>
						</div>
					{/if}
					<div class="absolute bottom-0 left-0 right-0 bg-black/60 text-white p-1">
						{#if item.size}
							<p class="text-xs opacity-80">{formatFileSize(item.size)}</p>
						{/if}
					</div>
				</div>
			{:else}
				<div class="bg-muted border-border flex items-center gap-2 rounded-lg border p-2">
					<div class="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded text-xs font-medium">
						{item.type.split('/').pop()?.toUpperCase() || 'FILE'}
					</div>
					<div class="flex flex-col">
						<span class="text-foreground text-sm font-medium truncate max-w-48">{item.name}</span>
						{#if item.size}
							<span class="text-muted-foreground text-xs">{formatFileSize(item.size)}</span>
						{/if}
					</div>
					{#if !readonly}
						<Button
							type="button"
							variant="ghost"
							size="sm"
							class="h-6 w-6 p-0"
							onclick={() => onFileRemove?.(item.id)}
						>
							<X class="h-3 w-3" />
						</Button>
					{/if}
				</div>
			{/if}
		{/each}
	</div>
{/if}
