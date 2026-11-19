<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { Mic, MicOff, Square } from '@lucide/svelte';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';

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
				class="size-8 max-md:size-9 rounded-full p-0 {isRecording
					? 'matte-orb orb-danger animate-pulse text-white! hover:bg-transparent'
					: showWarning
						? 'bg-surface-warning text-fg-warning hover:bg-surface-warning'
						: ''}"
				disabled={disabled || isLoading || !hasAudioModality}
				onclick={onMicClick}
				type="button"
				variant={isRecording || showWarning ? 'ghost' : 'default'}
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
