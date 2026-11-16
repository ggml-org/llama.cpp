<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { Mic, MicOff, Square } from '@lucide/svelte';
	import { ICON_CLASS_DEFAULT, VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';

	interface Props {
		/** What a press does: send a voice message or dictate. */
		action?: string;
		class?: string;
		disabled?: boolean;
		/** Said after the action in the tooltip, e.g. that another model may be loaded. */
		hint?: string | null;
		isLoading?: boolean;
		isRecording?: boolean;
		/** Why the microphone can't record (no HTTPS, access denied...), shown on the button. */
		micWarning?: string | null;
		/** Why no model can take the recording; disables the button. */
		unavailableReason?: string | null;
		onMicClick?: () => void;
	}

	let {
		action = VOICE_MODE_MESSAGE.voiceMessage,
		class: className = '',
		disabled = false,
		hint = null,
		isLoading = false,
		isRecording = false,
		micWarning = null,
		onMicClick,
		unavailableReason = null
	}: Props = $props();

	// no model to take the recording disables the button; a microphone that can't record keeps it
	// tappable, so a tap can say why
	const showWarning = $derived(!unavailableReason && !isRecording && !!micWarning);
	// a mouse starts recording on press, not on release, so the first words are not lost;
	// a touch or a key goes through the click, which is when the browser allows it
	const PRESS_CLICK_MS = 1000;
	let pressedAt = -Infinity;

	function handlePointerDown(event: PointerEvent) {
		if (isRecording || disabled || isLoading || unavailableReason) return;

		if (event.pointerType !== 'mouse' || event.button !== 0) return;

		pressedAt = performance.now();
		onMicClick?.();
	}

	function handleClick() {
		// the click that ends the press that already started it; the recording bar may have
		// taken the button's place meanwhile, so this one can also never come
		if (performance.now() - pressedAt < PRESS_CLICK_MS) {
			pressedAt = -Infinity;

			return;
		}

		onMicClick?.();
	}

	const tooltip = $derived(
		isRecording
			? null
			: (unavailableReason ??
					(showWarning ? micWarning : [action, hint].filter(Boolean).join('. ')))
	);
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				class="size-8 max-md:size-9 rounded-full p-0 {isRecording
					? 'matte-orb orb-danger text-white! hover:bg-transparent'
					: showWarning
						? 'bg-surface-warning text-fg-warning hover:bg-surface-warning'
						: ''}"
				disabled={disabled || isLoading || !!unavailableReason}
				onclick={handleClick}
				onpointerdown={handlePointerDown}
				type="button"
				variant={isRecording || showWarning ? 'ghost' : 'default'}
			>
				<span class="sr-only"
					>{isRecording
						? 'Stop recording'
						: showWarning
							? `Recording unavailable: ${micWarning}`
							: action}</span
				>

				{#if isRecording}
					<Square class="{ICON_CLASS_DEFAULT} fill-white" />
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
