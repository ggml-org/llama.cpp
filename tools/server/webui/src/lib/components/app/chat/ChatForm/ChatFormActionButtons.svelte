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
		variant="ghost"
		class="text-muted-foreground hover:text-foreground h-8 w-8 rounded-full p-0"
		disabled={disabled || isLoading}
		onclick={onFileUpload}
	>
		<Paperclip class="h-4 w-4" />
	</Button>

	<div>
		{#if isLoading}
			<Button
				type="button"
				variant="ghost"
				onclick={onStop}
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
				onclick={onMicClick}
			>
				<Mic class="h-8 w-8" />
			</Button>
			<Button
				type="submit"
				disabled={!canSend || disabled || isLoading}
				class="h-8 w-8 rounded-full p-0"
			>
				<ArrowUp class="h-12 w-12" />
			</Button>
		{/if}
	</div>
</div>
