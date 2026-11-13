<script lang="ts">
	import { Mic, MicOff, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		/** Why the microphone can't record (no HTTPS, access denied...), shown on the button. */
		micWarning?: string | null;
		onMicClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		isLoading = false,
		isRecording = false,
		micWarning = null,
		onMicClick
	}: Props = $props();

	// a model without audio input disables the button; a microphone that can't record keeps it
	// tappable, so a tap can say why
	const showWarning = $derived(hasAudioModality && !isRecording && !!micWarning);
	const tooltip = $derived(
		!hasAudioModality ? 'Current model does not support audio' : showWarning ? micWarning : null
	);
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				class="h-8 w-8 max-md:h-9 max-md:w-9 rounded-full p-0 {isRecording
					? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
					: showWarning
						? 'bg-amber-500/15 text-amber-600 hover:bg-amber-500/25 dark:text-amber-400'
						: ''}"
				disabled={disabled || isLoading || !hasAudioModality}
				onclick={onMicClick}
				type="button"
			>
				<span class="sr-only"
					>{isRecording
						? 'Stop recording'
						: showWarning
							? `Recording unavailable: ${micWarning}`
							: 'Start recording'}</span
				>

				{#if isRecording}
					<Square class="{ICON_CLASS_DEFAULT} animate-pulse fill-white" />
				{:else if showWarning}
					<MicOff class={ICON_CLASS_DEFAULT} />
				{:else}
					<Mic class={ICON_CLASS_DEFAULT} />
				{/if}
			</Button>
		</Tooltip.Trigger>

		{#if tooltip}
			<Tooltip.Content class="max-w-64">
				<p>{tooltip}</p>
			</Tooltip.Content>
		{/if}
	</Tooltip.Root>
</div>
