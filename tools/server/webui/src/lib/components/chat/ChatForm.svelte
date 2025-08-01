<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { Square, Paperclip, Mic, ArrowUp, Upload, X } from '@lucide/svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onSend?: (message: string) => void;
		onStop?: () => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: (files: File[]) => void;
		onFileRemove?: (fileId: string) => void;
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onSend,
		onStop,
		showHelperText = true,
		uploadedFiles = [],
		onFileUpload,
		onFileRemove
	}: Props = $props();

	let message = $state('');
	let textareaElement: HTMLTextAreaElement | undefined;
	let fileInputElement: HTMLInputElement | undefined;

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if (!message.trim() || disabled || isLoading) return;

		onSend?.(message.trim());
		message = '';

		if (textareaElement) {
			textareaElement.style.height = 'auto';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();

			if (!message.trim() || disabled || isLoading) return;

			onSend?.(message.trim());
			message = '';

			if (textareaElement) {
				textareaElement.style.height = 'auto';
			}
		}
	}

	function handleStop() {
		onStop?.();
	}

	function handleFileUpload() {
		fileInputElement?.click();
	}

	function handleFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files) {
			onFileUpload?.(Array.from(input.files));
		}
	}

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
</script>

<!-- Hidden file input -->
<input
	bind:this={fileInputElement}
	type="file"
	multiple
	accept="image/*,audio/*,video/*,.pdf,.txt,.doc,.docx"
	onchange={handleFileSelect}
	class="hidden"
/>

<form
	onsubmit={handleSubmit}
	class="border bg-muted/30 border-border/40 focus-within:border-primary/40 bg-background dark:bg-muted border-radius-bottom-none mx-auto max-w-4xl overflow-hidden rounded-3xl {className}"
>
	<!-- File previews -->
	{#if uploadedFiles.length > 0}
		<div class="mb-3 flex flex-wrap items-start gap-3 px-5 pt-3">
			{#each uploadedFiles as file (file.id)}
				{#if file.preview}
					<!-- Image file with thumbnail -->
					<div class="bg-muted border-border relative rounded-lg border overflow-hidden">
						<img 
							src={file.preview} 
							alt={file.name} 
							class="h-24 w-24 object-cover" 
						/>
						<div class="absolute top-1 right-1 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
							<Button
								type="button"
								variant="ghost"
								size="sm"
								class="h-6 w-6 p-0 bg-white/20 hover:bg-white/30 text-white"
								onclick={() => onFileRemove?.(file.id)}
							>
								<X class="h-3 w-3" />
							</Button>
						</div>
						<div class="absolute bottom-0 left-0 right-0 bg-black/60 text-white p-1">
							<p class="text-xs opacity-80">{formatFileSize(file.size)}</p>
						</div>
					</div>
				{:else}
					<!-- Non-image file with badge -->
					<div class="bg-muted border-border flex items-center gap-2 rounded-lg border p-2">
						<div class="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded text-xs font-medium">
							{file.name.split('.').pop()?.toUpperCase() || 'FILE'}
						</div>
						<div class="flex flex-col">
							<span class="text-foreground text-sm font-medium truncate max-w-48">{file.name}</span>
							<span class="text-muted-foreground text-xs">{formatFileSize(file.size)}</span>
						</div>
						<Button
							type="button"
							variant="ghost"
							size="sm"
							class="h-6 w-6 p-0"
							onclick={() => onFileRemove?.(file.id)}
						>
							<X class="h-3 w-3" />
						</Button>
					</div>
				{/if}
			{/each}
		</div>
	{/if}

	<div
		class="flex-column relative min-h-[48px] items-center rounded-3xl px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
	>
		<div class="flex-1">
			<textarea
				bind:this={textareaElement}
				bind:value={message}
				onkeydown={handleKeydown}
				oninput={(event) => autoResizeTextarea(event.currentTarget)}
				placeholder="Ask anything..."
				class="placeholder:text-muted-foreground text-md max-h-32 min-h-[24px] w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
				{disabled}
			></textarea>
		</div>

		<div class="flex items-center justify-between gap-1">
			<Button
				type="button"
				variant="ghost"
				class="text-muted-foreground hover:text-foreground h-8 w-8 rounded-full p-0"
				disabled={disabled || isLoading}
				onclick={handleFileUpload}
			>
				<Paperclip class="h-4 w-4" />
			</Button>

			<div>
				{#if isLoading}
					<Button
						type="button"
						variant="ghost"
						onclick={handleStop}
						class="text-muted-foreground hover:text-destructive h-8 w-8 rounded-full p-0"
					>
						<Square class="h-8 w-8" />
					</Button>
				{:else}
					<Button
						type="button"
						variant="ghost"
						class="text-muted-foreground hover:text-foreground h-8 w-8 rounded-full p-0"
						disabled={disabled || isLoading}
					>
						<Mic class="h-8 w-8" />
					</Button>
					<Button
						type="submit"
						disabled={!message.trim() || disabled || isLoading}
						class="h-8 w-8 rounded-full p-0"
					>
						<ArrowUp class="h-12 w-12" />
					</Button>
				{/if}
			</div>
		</div>
	</div>
</form>

{#if showHelperText}
	<div class="mt-4 flex items-center justify-center">
		<p class="text-muted-foreground text-xs">
			Press <kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Enter</kbd> to send,
			<kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new
			line
		</p>
	</div>
{/if}
