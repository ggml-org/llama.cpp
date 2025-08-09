<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Square, Paperclip, Mic, ArrowUp } from '@lucide/svelte';

	interface Props {
		disabled?: boolean;
		isLoading?: boolean;
		canSend?: boolean;
		onFileUpload?: () => void;
		onStop?: () => void;
		onMicClick?: () => void;
		class?: string;
	}

	let {
		disabled = false,
		isLoading = false,
		canSend = false,
		onFileUpload,
		onStop,
		onMicClick,
		class: className = ''
	}: Props = $props();
</script>

<div class="flex items-center justify-between gap-1 {className}">
	<Button
		type="button"
		class="text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground h-8 w-8 rounded-full p-0"
		disabled={disabled || isLoading}
		onclick={onFileUpload}
	>
		<span class="sr-only">Attach files</span>
		<Paperclip class="h-4 w-4" />
	</Button>

	<div class="flex gap-2">
		{#if isLoading}
			<Button
				type="button"
				onclick={onStop}
				class="p-0 h-8 w-8 bg-transparent hover:bg-destructive/20"
			>
				<span class="sr-only">Stop</span>
				<Square class="h-8 w-8 fill-destructive stroke-destructive" />
			</Button>
		{:else}
			<Button
				type="button"
				class="text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground h-8 w-8 rounded-full p-0"
				disabled={disabled || isLoading}
				onclick={onMicClick}
			>
				<span class="sr-only">Start recording</span>
				<Mic class="h-8 w-8" />
			</Button>

			<Button
				type="submit"
				disabled={!canSend || disabled || isLoading}
				class="h-8 w-8 rounded-full p-0"
			>
				<span class="sr-only">Send</span>
				<ArrowUp class="h-12 w-12" />
			</Button>
		{/if}
	</div>
</div>
