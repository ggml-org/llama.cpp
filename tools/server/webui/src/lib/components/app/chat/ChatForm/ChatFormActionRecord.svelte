<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Mic } from '@lucide/svelte';
	import { supportsAudio } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		onMicClick?: () => void;
		class?: string;
	}

	let {
		disabled = false,
		isLoading = false,
		isRecording = false,
		onMicClick,
		class: className = ''
	}: Props = $props();
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root delayDuration={100}>
		<Tooltip.Trigger>
			<Button
				type="button"
				class="h-8 w-8 rounded-full p-0 {isRecording
					? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
					: 'text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground'} {!supportsAudio()
					? 'opacity-50 cursor-not-allowed'
					: ''}"
				disabled={disabled || isLoading || !supportsAudio()}
				onclick={onMicClick}
			>
				<span class="sr-only">{isRecording ? 'Stop recording' : 'Start recording'}</span>
				<Mic class="h-4 w-4" />
			</Button>
		</Tooltip.Trigger>

		{#if !supportsAudio()}
			<Tooltip.Content>
				<p>Current model does not support audio</p>
			</Tooltip.Content>
		{/if}
	</Tooltip.Root>
</div>
