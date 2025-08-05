<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { Square, Paperclip, Mic, ArrowUp, Upload, X } from '@lucide/svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';
	import { ChatAttachmentsList } from '$lib/components';
	import { inputClasses } from '$lib/constants/input-classes';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: (files: File[]) => void;
		onFileRemove?: (fileId: string) => void;
		pasteLongTextToFileLength?: number;
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onSend,
		onStop,
		showHelperText = true,
		uploadedFiles = $bindable([]),
		onFileUpload,
		onFileRemove,
		pasteLongTextToFileLength = 2500
	}: Props = $props();

	let message = $state('');
	let textareaElement: HTMLTextAreaElement | undefined;
	let fileInputElement: HTMLInputElement | undefined;
	let previousIsLoading = $state(isLoading);

	onMount(() => {
		if (textareaElement) {
			textareaElement.focus();
		}
	});

	$effect(() => {
		if (previousIsLoading && !isLoading && textareaElement) {
			textareaElement.focus();
		}

		previousIsLoading = isLoading;
	});

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if (!message.trim() || disabled || isLoading) return;

		// Store the message and files before clearing
		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		// Clear the form immediately to hide message and attachments
		message = '';
		uploadedFiles = [];

		if (textareaElement) {
			textareaElement.style.height = 'auto';
		}

		// Send the message with the stored data
		const success = await onSend?.(messageToSend, filesToSend);

		// If sending failed, restore the form state
		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
		}
	}

	async function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();

			if (!message.trim() || disabled || isLoading) return;

			// Store the message and files before clearing
			const messageToSend = message.trim();
			const filesToSend = [...uploadedFiles];

			// Clear the form immediately to hide message and attachments
			message = '';
			uploadedFiles = [];

			if (textareaElement) {
				textareaElement.style.height = 'auto';
			}

			// Send the message with the stored data
			const success = await onSend?.(messageToSend, filesToSend);

			// If sending failed, restore the form state
			if (!success) {
				message = messageToSend;
				uploadedFiles = filesToSend;
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

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFileUpload?.(files);
			return;
		}

		const text = event.clipboardData.getData('text/plain');
		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: 'text/plain'
			});

			onFileUpload?.([textFile]);
		}
	}
</script>

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
	class="{inputClasses} border-radius-bottom-none mx-auto max-w-4xl overflow-hidden rounded-3xl backdrop-blur-md {className}"
>
	<ChatAttachmentsList bind:uploadedFiles {onFileRemove} class="mb-3 px-5 pt-5" />

	<div
		class="flex-column relative min-h-[48px] items-center rounded-3xl px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
		onpaste={handlePaste}
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
			<kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new line
		</p>
	</div>
{/if}
