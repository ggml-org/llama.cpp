<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Textarea } from '$lib/components/ui/textarea';
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { Send, Square } from '@lucide/svelte';

	let {
		disabled = false,
		isLoading = false,
		onsend,
		onstop
	}: {
		disabled?: boolean;
		isLoading?: boolean;
		onsend?: (message: string) => void;
		onstop?: () => void;
	} = $props();

	let message = $state('');
	let textareaElement = $state<HTMLTextAreaElement>();

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

<div class="bg-background border-t p-4">
	<form onsubmit={handleSubmit} class="flex items-center gap-4">
		<div class="flex-1">
			<Textarea
				bind:value={message}
				onkeydown={handleKeydown}
				oninput={(event) => autoResizeTextarea(event.currentTarget)}
				placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
				class="max-h-32 min-h-[44px] resize-none"
				{disabled}
			/>
		</div>

		{#if isLoading}
			<Button
				type="button"
				variant="outline"
				size="icon"
				onclick={handleStop}
				class="h-12 w-12 shrink-0"
			>
				<Square class="h-4 w-4" />
			</Button>
		{:else}
			<Button
				type="submit"
				size="icon"
				disabled={!message.trim() || disabled}
				class="h-14 w-14 shrink-0"
			>
				<Send class="h-4 w-4" />
			</Button>
		{/if}
	</form>
</div>
