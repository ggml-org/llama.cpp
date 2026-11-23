<!--
	An audio attachment drawn as a clip: a playable waveform, and for a voice message its transcript under it, open while
	it comes in and after, unless the Collapse transcripts setting folds it (then it opens on hover, keyboard focus or a
	tap). `sm` is the chip in the chat form, `md` the clip in a sent message. A clip
	stored without peaks (an older message, an uploaded file) gets them computed from its audio.
-->
<script lang="ts">
	import {
		spokenLength,
		VoiceTranscriptReveal,
		VoiceWaveform,
		VoiceWaveformSize
	} from '@ggml-org/llama-ui/voice';
	import { X } from '@lucide/svelte';
	import { ActionIcon } from '#lib/components/app/index.ts';
	import { SETTINGS_KEYS, VOICE_MESSAGE_LABEL } from '#lib/constants/index.ts';
	import { settingsStore, voiceTranscriptsStore } from '#lib/stores/index.ts';
	import type { ChatAttachmentDisplayItem } from '#lib/types/index.ts';
	import { audioUrl, lazyPeaks, voiceAttachmentSource } from '#lib/utils/voice-attachment.ts';

	interface Props {
		class?: string;
		item: ChatAttachmentDisplayItem;
		onRemove?: (id: string) => void;
		readonly?: boolean;
		size?: VoiceWaveformSize;
	}

	let {
		class: className = '',
		item,
		onRemove,
		readonly = false,
		size = VoiceWaveformSize.MD
	}: Props = $props();

	let source = $derived(voiceAttachmentSource(item));
	let computedPeaks = $state<number[] | null>(null);
	let src = $state<string | undefined>();

	let peaks = $derived(source.peaks ?? computedPeaks ?? []);
	let shown = $derived(voiceTranscriptsStore.shown(source.name, source.transcript));
	let transcript = $derived(shown.transcript);
	let status = $derived(shown.status);
	let errorReason = $derived(shown.errorReason);
	let label = $derived(source.voice ? VOICE_MESSAGE_LABEL : source.name);
	// the transcript lights up word by word as the clip plays, and stays lit
	let playbackMs = $state<number | null>(null);
	let spoken = $derived(
		transcript && playbackMs !== null && source.durationMs
			? spokenLength(transcript, peaks, source.durationMs, playbackMs)
			: null
	);
	let collapsed = $derived(settingsStore.config[SETTINGS_KEYS.VOICE_COLLAPSE_TRANSCRIPTS] === true);

	// an object URL lives as long as the clip is on screen
	$effect(() => {
		const audio = source.audio;

		if (!audio) {
			src = undefined;

			return;
		}

		const playable = audioUrl(audio, source.mimeType);

		src = playable?.url;

		return () => {
			if (playable?.revoke) URL.revokeObjectURL(playable.url);
		};
	});

	$effect(() => {
		const { audio, name, peaks: stored } = source;

		computedPeaks = null;

		if (stored || !audio) return;

		let cancelled = false;

		lazyPeaks(name, audio).then((value) => {
			if (!cancelled) computedPeaks = value;
		});

		return () => {
			cancelled = true;
		};
	});
</script>

<div
	class={[
		'group relative min-w-0',
		size === VoiceWaveformSize.SM
			? 'w-60 max-w-[calc(100vw-4rem)] rounded-lg border border-border bg-muted py-2 pr-9 pl-2'
			: 'w-80 max-w-full rounded-2xl matte-sheen bg-(--user-bubble-background) px-3 py-2.5 backdrop-blur-md dark:bg-primary/15',
		className
	]}
	data-slot="chat-attachment-voice"
	data-voice={source.voice || undefined}
>
	{#key collapsed}
		<VoiceTranscriptReveal defaultOpen={!collapsed} {errorReason} {spoken} {status} {transcript}>
			<VoiceWaveform bind:playbackMs durationMs={source.durationMs} {label} {peaks} {size} {src} />
		</VoiceTranscriptReveal>
	{/key}

	{#if !source.voice}
		<p class="mt-1 truncate px-1 text-caption text-muted-foreground">{source.name}</p>
	{/if}

	{#if !readonly && onRemove}
		<div class="absolute top-1/2 right-1 -translate-y-1/2">
			<ActionIcon
				ariaLabel={source.voice ? 'Remove voice message' : `Remove ${source.name}`}
				icon={X}
				onclick={() => onRemove?.(item.id)}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>
