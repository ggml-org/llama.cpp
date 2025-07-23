<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Textarea } from '$lib/components/ui/textarea';
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { Send, Square, Paperclip, Mic } from '@lucide/svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onsend?: (message: string) => void;
		onstop?: () => void;
		showHelperText?: boolean;
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onsend,
		onstop,
		showHelperText = true
	}: Props = $props();

	let message = $state('');
	let textareaElement: HTMLTextAreaElement | undefined;

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if (!message.trim() || disabled || isLoading) return;

		onsend?.(message.trim());
		message = '';

		if (textareaElement) {
			textareaElement.style.height = 'auto';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();

			if (!message.trim() || disabled || isLoading) return;

			onsend?.(message.trim());
			message = '';

			if (textareaElement) {
				textareaElement.style.height = 'auto';
			}
		}
	}

	function handleStop() {
		onstop?.();
	}
</script>

<div class="bg-background p-4 {className}">
	<form onsubmit={handleSubmit} class="mx-auto max-w-4xl">
		<!-- Input Container -->
		<div
			class="bg-muted/30 border-border/40 focus-within:border-primary/40 flex-column relative min-h-[48px] items-center rounded-3xl border px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
		>
			<!-- Text Input -->
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

			<!-- Actions Bar -->
			<div class="flex items-center justify-between gap-1">
				<!-- Left Actions -->
				<Button
					type="button"
					variant="ghost"
					class="text-muted-foreground hover:text-foreground h-9 w-9 rounded-full p-0"
					disabled={disabled || isLoading}
				>
					<Paperclip class="h-4 w-4" />
				</Button>

				<div>
					{#if isLoading}
						<Button
							type="button"
							variant="ghost"
							onclick={handleStop}
							class="text-muted-foreground hover:text-destructive h-9 w-9 rounded-full p-0"
						>
							<Square class="h-6 w-6" />
						</Button>
					{:else}
						<Button
							type="button"
							variant="ghost"
							class="text-muted-foreground hover:text-foreground h-9 w-9 rounded-full p-0"
							disabled={disabled || isLoading}
						>
							<Mic class="h-6 w-6" />
						</Button>
						<Button
							type="submit"
							disabled={!message.trim() || disabled || isLoading}
							class="h-9 w-9 rounded-full p-0"
						>
							<Send class="h-6 w-6" />
						</Button>
					{/if}
				</div>
			</div>
		</div>

		<!-- Helper Text -->
		{#if showHelperText}
			<div class="mt-2 flex items-center justify-center">
				<p class="text-muted-foreground text-xs">
					Press <kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Enter</kbd> to
					send,
					<kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for
					new line
				</p>
			</div>
		{/if}
	</form>
</div>
