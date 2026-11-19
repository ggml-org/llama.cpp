<!--
	The chat form's actions row while it records: Llama UI's VoiceRecordingBar with the form's labels and its
	microphone button as the stop button. A voice message's text, when the model streams it, runs under the waveform;
	dictated text goes into the message instead. After a dictation that streamed nothing, it waits for the text in
	place of the waveform. Speech that stays too quiet even boosted gets a line under the waveform saying so.
-->
<script lang="ts">
	import { VoiceRecordingBar } from '@ggml-org/llama-ui/voice';
	import { ChatFormActionRecord } from '#lib/components/app/index.ts';
	import { VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
	import { VoiceMode } from '#lib/enums/index.ts';
	import type { ChatFormVoiceState } from '#lib/hooks/use-chat-form-voice.svelte.ts';

	interface Props {
		class?: string;
		elapsedMs: number;
		levels: number[];
		/** Text streaming while a voice message is recorded. */
		liveCaption?: string;
		mode: VoiceMode;
		/** The speech is too quiet even at the highest gain: says so under the waveform. */
		quiet?: boolean;
		state: ChatFormVoiceState;
		onCancel: () => void;
		onStop: () => void;
		/** Stops and sends; left out, there is no send button. */
		onSend?: () => void;
	}

	let {
		class: className = '',
		elapsedMs,
		levels,
		liveCaption = '',
		mode,
		onCancel,
		onSend,
		onStop,
		quiet = false,
		state
	}: Props = $props();

	const LABELS = {
		starting: VOICE_MODE_MESSAGE.starting,
		waiting: VOICE_MODE_MESSAGE.transcribing
	};
</script>

<VoiceRecordingBar
	caption={liveCaption}
	class={className}
	{elapsedMs}
	label={mode === VoiceMode.DICTATION ? VOICE_MODE_MESSAGE.dictating : VOICE_MODE_MESSAGE.recording}
	labels={LABELS}
	{levels}
	{onCancel}
	{onSend}
	status={state === 'transcribing' ? 'waiting' : state === 'starting' ? 'starting' : 'recording'}
	warning={quiet ? VOICE_MODE_MESSAGE.tooQuiet : null}
>
	{#snippet stop({ disabled })}
		<ChatFormActionRecord class="shrink-0" {disabled} isRecording onMicClick={onStop} />
	{/snippet}
</VoiceRecordingBar>
