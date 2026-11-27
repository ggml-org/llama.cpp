<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { Mic, MicOff, Square } from '@lucide/svelte';
	import {
		ICON_CLASS_DEFAULT,
		VOICE_MODE_MESSAGE,
		VOICE_PRESS_CLICK_MS
	} from '#lib/constants/index.ts';
	import { MouseButton, PointerType } from '#lib/enums/index.ts';

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
		/** A quieter (tertiary) button, beside the main one: send or stop. */
		secondary?: boolean;
		/** With `secondary`: filled like the stop button it stands beside, not bare. */
		filled?: boolean;
		onMicClick?: () => void;
		/**
		 * A mouse or the focus is on the button: a press may follow, the microphone can open
		 * once it rests there. Called again as the pointer moves on it.
		 */
		onPrepare?: () => void;
		/** It left again without a press. */
		onPrepareEnd?: () => void;
	}

	let {
		action = VOICE_MODE_MESSAGE.voiceMessage,
		class: className = '',
		disabled = false,
		filled = false,
		hint = null,
		isLoading = false,
		isRecording = false,
		micWarning = null,
		onMicClick,
		onPrepare,
		onPrepareEnd,
		secondary = false,
		unavailableReason = null
	}: Props = $props();

	// no model to take the recording disables the button; a microphone that can't record keeps it
	// tappable, so a tap can say why
	const showWarning = $derived(!unavailableReason && !isRecording && !!micWarning);
	// a mouse starts recording on press, not on release, so the first words are not lost;
	// a touch or a key goes through the click, which is when the browser allows it
	let pressedAt = -Infinity;

	function handlePointerDown(event: PointerEvent) {
		if (isRecording || disabled || isLoading || unavailableReason) return;

		if (event.pointerType !== PointerType.MOUSE || event.button !== MouseButton.PRIMARY) return;

		pressedAt = performance.now();
		onMicClick?.();
	}

	// a touch presses at once, so only a mouse, a pen or the keyboard come ahead of the press
	function handlePointerEnter(event: PointerEvent) {
		if (event.pointerType !== PointerType.TOUCH && !isRecording) onPrepare?.();
	}

	// the pointer still moving over the button is passing, so the wait for it to rest restarts
	function handlePointerMove(event: PointerEvent) {
		if (event.pointerType !== PointerType.TOUCH && !isRecording) onPrepare?.();
	}

	function handlePointerLeave(event: PointerEvent) {
		if (event.pointerType !== PointerType.TOUCH) onPrepareEnd?.();
	}

	function handleClick() {
		// the click that ends the press that already started it; the recording bar may have
		// taken the button's place meanwhile, so this one can also never come
		if (performance.now() - pressedAt < VOICE_PRESS_CLICK_MS) {
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
				class="group size-8 max-md:size-9 rounded-full p-0 {isRecording
					? 'matte-orb orb-danger text-white! hover:bg-transparent'
					: showWarning
						? 'bg-surface-warning text-fg-warning hover:bg-surface-warning'
						: ''}"
				disabled={disabled || isLoading || !!unavailableReason}
				onblur={() => onPrepareEnd?.()}
				onclick={handleClick}
				onfocus={() => !isRecording && onPrepare?.()}
				onpointerdown={handlePointerDown}
				onpointerenter={handlePointerEnter}
				onpointerleave={handlePointerLeave}
				onpointermove={handlePointerMove}
				type="button"
				variant={isRecording || showWarning
					? 'ghost'
					: secondary
						? filled
							? 'secondary'
							: 'tertiary'
						: 'default'}
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
				{:else if secondary}
					<Mic class="{ICON_CLASS_DEFAULT} stroke-muted-foreground group-hover:stroke-foreground" />
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
