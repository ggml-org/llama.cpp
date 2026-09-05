<script lang="ts">
	import { LoaderCircle, Mic, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		isTranscribing?: boolean;
		onMicClick?: () => void;
		transcriptionModelName?: string | null;
	}

	let {
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		isTranscribing = false,
		onMicClick,
		transcriptionModelName = null
	}: Props = $props();
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				class="h-8 w-8 rounded-full p-0 {isRecording
					? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
					: ''}"
				disabled={disabled || isLoading || isTranscribing}
				onclick={onMicClick}
				type="button"
			>
				<span class="sr-only">{isRecording ? 'Stop recording' : 'Start recording'}</span>

				{#if isTranscribing}
					<LoaderCircle class="{ICON_CLASS_DEFAULT} animate-spin" />
				{:else if isRecording}
					<Square class="{ICON_CLASS_DEFAULT} animate-pulse fill-white" />
				{:else}
					<Mic class={ICON_CLASS_DEFAULT} />
				{/if}
			</Button>
		</Tooltip.Trigger>

		{#if transcriptionModelName}
			<Tooltip.Content>
				<p>Voice input is transcribed by {transcriptionModelName}</p>
			</Tooltip.Content>
		{/if}
	</Tooltip.Root>
</div>
