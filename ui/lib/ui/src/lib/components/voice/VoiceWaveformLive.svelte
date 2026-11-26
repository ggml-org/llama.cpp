<!--
	The waveform while recording, sized to sit in the chat form's input row and always filling its width: a short
	recording is stretched over all the bars, a longer one scrolls with the newest level on the right. The app passes
	the levels; this only draws them.
	Decorative: hidden from screen readers, with an optional visually hidden `label` (e.g. `Recording`) in its place.
	With reduced motion the bars stay still.
-->
<script lang="ts">
	import { prefersReducedMotion } from 'svelte/motion';
	import { cn } from '../ui/utils';
	import { VOICE_BAR_GAP_PX, VOICE_LIVE_BAR_WIDTH_PX } from '../../constants/voice.constants';
	import { barCapacity, barHeightPercent, liveBars } from '../../voice/waveform';

	interface Props {
		/** Input levels, 0..1, newest last. */
		levels: number[];
		/** Off when paused or not listening: the bars turn muted. */
		active?: boolean;
		/** Read by screen readers in place of the bars, e.g. `Recording`. */
		label?: string;
		class?: string;
	}

	let { active = true, class: className, label, levels }: Props = $props();

	let width = $state(0);

	let bars = $derived(
		liveBars(prefersReducedMotion.current ? [] : levels, barCapacity(width, VOICE_LIVE_BAR_WIDTH_PX, VOICE_BAR_GAP_PX))
	);
</script>

<div
	class={cn('relative h-6 w-full min-w-0', active ? 'text-foreground' : 'text-muted-foreground', className)}
	data-slot="voice-waveform-live"
>
	{#if label}
		<span class="sr-only">{label}</span>
	{/if}

	<div
		style:gap="{VOICE_BAR_GAP_PX}px"
		class="flex h-full items-center justify-end overflow-hidden"
		aria-hidden="true"
		bind:clientWidth={width}
	>
		{#each bars as level, index (index)}
			<span
				style:width="{VOICE_LIVE_BAR_WIDTH_PX}px"
				style:height="{barHeightPercent(level)}%"
				class={[
					'shrink-0 rounded-full bg-current transition-[height] duration-100 ease-linear motion-reduce:transition-none',
					level === null && 'opacity-40'
				]}
			></span>
		{/each}
	</div>
</div>
