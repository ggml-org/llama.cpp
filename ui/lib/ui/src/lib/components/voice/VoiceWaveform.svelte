<!--
	A recorded voice clip: a play button, its waveform with the played part filled in, and the time. `sm` is the chip in
	the chat form, `md` the clip in a message bubble; the surface around it is the caller's.
	The waveform is a slider: click or drag to seek, arrow keys move 5 s. Playback uses its own <audio>, and starting
	one clip pauses any other. Without `src` it shows the waveform and the length only.
-->
<script lang="ts" module>
	// one clip plays at a time: starting one pauses the one before
	let playingAudio: HTMLAudioElement | null = null;
</script>

<script lang="ts">
	import { Pause, Play } from '@lucide/svelte';
	import { onDestroy } from 'svelte';
	import { Button } from '../ui/button';
	import { cn } from '../ui/utils';
	import {
		VOICE_BAR_GAP_PX,
		VOICE_SEEK_STEP_MS,
		VOICE_WAVEFORM_DEFAULT_LABEL,
		VOICE_WAVEFORM_SIZES
	} from '../../constants/voice.constants';
	import { KeyboardKey, MouseButton } from '../../enums/ui.enums';
	import { VoiceWaveformSize } from '../../enums/voice.enums';
	import {
		barCapacity,
		barHeightPercent,
		formatClipPosition,
		formatClipTime,
		pointerFraction,
		progressFraction,
		resamplePeaks,
		stepPosition
	} from '../../voice/waveform';

	interface Props {
		/** The clip's peaks, 0..1, resampled to the bars that fit. */
		peaks: number[];
		/** The audio to play: an object URL or a data URL. */
		src?: string;
		/** The clip's length; read from the audio when left out. */
		durationMs?: number;
		size?: VoiceWaveformSize;
		/** What the clip is, for the play button and the slider. */
		label?: string;
		/**
		 * Bound out: where playback is, to follow a transcript with. Null until the clip first plays, the clip's end
		 * once it played through.
		 */
		playbackMs?: number | null;
		class?: string;
	}

	let {
		class: className,
		durationMs,
		label = VOICE_WAVEFORM_DEFAULT_LABEL,
		peaks,
		playbackMs = $bindable(null),
		size = VoiceWaveformSize.MD,
		src
	}: Props = $props();

	let audio = $state<HTMLAudioElement>();
	let track = $state<HTMLDivElement>();
	let paused = $state(true);
	let currentTime = $state(0);
	let mediaDuration = $state(0);
	let failed = $state(false);
	let dragging = $state(false);
	let width = $state(0);

	let sized = $derived(VOICE_WAVEFORM_SIZES[size]);
	let playable = $derived(!!src && !failed);
	let totalMs = $derived(
		durationMs ?? (Number.isFinite(mediaDuration) ? Math.max(mediaDuration, 0) * 1000 : 0)
	);
	let elapsedMs = $derived(Math.min(currentTime * 1000, totalMs || Infinity));
	let progress = $derived(progressFraction(elapsedMs, totalMs));
	let bars = $derived(resamplePeaks(peaks, barCapacity(width, sized.barWidthPx, VOICE_BAR_GAP_PX)));
	// the length until playback starts, then where it is
	let shownMs = $derived(!paused || elapsedMs > 0 ? elapsedMs : totalMs);

	// playback as it goes, read every frame while playing: the media element's own updates come a few times a second
	let started = $state(false);
	let playedThrough = $state(false);

	$effect(() => {
		if (!started) {
			playbackMs = null;

			return;
		}

		if (playedThrough) {
			playbackMs = totalMs;

			return;
		}

		playbackMs = elapsedMs;

		if (paused || !audio) return;

		const media = audio;
		let frame = requestAnimationFrame(function tick() {
			playbackMs = Math.min(media.currentTime * 1000, totalMs || Infinity);
			frame = requestAnimationFrame(tick);
		});

		return () => cancelAnimationFrame(frame);
	});

	$effect(() => {
		if (!src) paused = true;

		failed = false;
		currentTime = 0;
	});

	onDestroy(() => {
		if (playingAudio === audio) playingAudio = null;
	});

	function toggle() {
		if (!audio || !playable) return;

		if (audio.paused) audio.play().catch(() => {});
		else audio.pause();
	}

	function seekTo(ms: number) {
		if (!audio || !playable) return;

		audio.currentTime = ms / 1000;
		currentTime = ms / 1000;
	}

	function seekToPointer(event: PointerEvent) {
		if (!track) return;

		const rect = track.getBoundingClientRect();

		seekTo(pointerFraction(event.clientX, rect.left, rect.width) * totalMs);
	}

	function handlePointerDown(event: PointerEvent) {
		if (!playable || event.button !== MouseButton.PRIMARY) return;

		try {
			track?.setPointerCapture(event.pointerId);
		} catch {
			// a scripted pointer has no capture to take; the press still seeks
		}

		dragging = true;
		seekToPointer(event);
	}

	function handlePointerMove(event: PointerEvent) {
		if (dragging) seekToPointer(event);
	}

	function handlePointerEnd(event: PointerEvent) {
		if (!dragging) return;

		dragging = false;

		if (track?.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
	}

	function handleKeydown(event: KeyboardEvent) {
		if (!playable) return;

		const target =
			event.key === KeyboardKey.ARROW_RIGHT || event.key === KeyboardKey.ARROW_UP
				? stepPosition(elapsedMs, VOICE_SEEK_STEP_MS, totalMs)
				: event.key === KeyboardKey.ARROW_LEFT || event.key === KeyboardKey.ARROW_DOWN
					? stepPosition(elapsedMs, -VOICE_SEEK_STEP_MS, totalMs)
					: event.key === KeyboardKey.HOME
						? 0
						: event.key === KeyboardKey.END
							? totalMs
							: null;

		if (target === null) return;

		event.preventDefault();
		seekTo(target);
	}

	function handlePlay() {
		if (playingAudio && playingAudio !== audio) playingAudio.pause();

		started = true;
		playedThrough = false;

		playingAudio = audio ?? null;
	}

	function handlePause() {
		if (playingAudio === audio) playingAudio = null;
	}

	function handleEnded() {
		playedThrough = true;

		if (audio) audio.currentTime = 0;

		currentTime = 0;
	}
</script>

{#snippet barRow(className: string)}
	<div style:gap="{VOICE_BAR_GAP_PX}px" class={cn('flex h-full items-center', className)} aria-hidden="true">
		{#each bars as peak, index (index)}
			<span
				style:width="{sized.barWidthPx}px"
				style:height="{barHeightPercent(peak)}%"
				class="shrink-0 rounded-full bg-current"
			></span>
		{/each}
	</div>
{/snippet}

<div
	class={cn('flex w-full min-w-0 items-center', sized.gap, className)}
	data-size={size}
	data-slot="voice-waveform"
>
	<Button
		class={cn('shrink-0 rounded-full p-0', sized.button)}
		aria-label={paused ? `Play ${label.toLowerCase()}` : `Pause ${label.toLowerCase()}`}
		disabled={!playable}
		onclick={toggle}
	>
		{#if paused}
			<Play
				class={cn('translate-x-px fill-current', sized.icon)}
			/>
		{:else}
			<Pause class={cn('fill-current', sized.icon)} />
		{/if}
	</Button>

	<div
		bind:this={track}
		class={cn(
			'relative min-w-0 flex-1 touch-none rounded-md outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/50',
			sized.track,
			playable ? 'cursor-pointer' : 'cursor-default'
		)}
		aria-disabled={!playable || undefined}
		aria-label="{label} position"
		aria-valuemax={Math.round(totalMs / 1000)}
		aria-valuemin={0}
		aria-valuenow={Math.round(elapsedMs / 1000)}
		aria-valuetext={formatClipPosition(elapsedMs, totalMs)}
		bind:clientWidth={width}
		onkeydown={handleKeydown}
		onlostpointercapture={() => (dragging = false)}
		onpointercancel={handlePointerEnd}
		onpointerdown={handlePointerDown}
		onpointermove={handlePointerMove}
		onpointerup={handlePointerEnd}
		role="slider"
		tabindex={playable ? 0 : -1}
	>
		{@render barRow('text-muted-foreground/40')}

		<div
			style:clip-path="inset(0 {100 - progress * 100}% 0 0)"
			class="absolute inset-0 text-primary"
		>
			{@render barRow('')}
		</div>
	</div>

	<span
		class={cn(
			'shrink-0 text-right text-caption text-muted-foreground tabular-nums',
			sized.time
		)}
		aria-hidden="true"
	>
		{formatClipTime(shownMs)}
	</span>

	{#if src}
		<audio
			bind:this={audio}
			bind:currentTime
			bind:duration={mediaDuration}
			bind:paused
			onended={handleEnded}
			onerror={() => (failed = true)}
			onpause={handlePause}
			onplay={handlePlay}
			preload="metadata"
			{src}
		></audio>
	{/if}
</div>
