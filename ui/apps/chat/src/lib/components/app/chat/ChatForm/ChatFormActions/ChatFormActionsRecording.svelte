<!--
	The chat form's actions row while it records: cancel, the live waveform and the time, stop, and send, which stops
	and sends what was recorded in one go. A voice message's
	text, when the model streams it, runs under the waveform; dictated text goes into the message instead. After a
	dictation that streamed nothing, it waits for the text in place of the waveform.
-->
<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { formatClipTime, VoiceSpokenText, VoiceWaveformLive } from '@ggml-org/llama-ui/voice';
	import { ArrowUp, LoaderCircle, X } from '@lucide/svelte';
	import { ChatFormActionRecord } from '#lib/components/app/index.ts';
	import { ICON_CLASS_DEFAULT, VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
	import { VoiceMode } from '#lib/enums/index.ts';
	import type { ChatFormVoiceState } from '#lib/hooks/use-chat-form-voice.svelte.ts';

	interface Props {
		class?: string;
		elapsedMs: number;
		levels: number[];
		/** Text streaming while a voice message is recorded. */
		liveCaption?: string;
		mode: VoiceMode;
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
		state
	}: Props = $props();
</script>

<div class={['flex w-full min-w-0 flex-col gap-1', className]} data-slot="chat-form-recording">
	<div class="flex w-full min-w-0 items-center gap-2">
		<Button
			class="size-8 shrink-0 rounded-full p-0 text-muted-foreground max-md:size-9"
			onclick={onCancel}
			type="button"
			variant="ghost"
		>
			<span class="sr-only">Cancel recording</span>

			<X class={ICON_CLASS_DEFAULT} />
		</Button>

		{#if state === 'transcribing'}
			<div
				class="flex min-w-0 flex-1 items-center gap-2 px-1 text-caption text-fg-muted"
				role="status"
			>
				<LoaderCircle class="size-icon-sm animate-spin motion-reduce:animate-none" />
				{VOICE_MODE_MESSAGE.transcribing}
			</div>
		{:else}
			<VoiceWaveformLive
				active={state === 'recording'}
				class="flex-1 px-1"
				label={mode === VoiceMode.DICTATION
					? VOICE_MODE_MESSAGE.dictating
					: VOICE_MODE_MESSAGE.recording}
				{levels}
			/>

			<!-- the timer starts with the first audio: until then, what is said is not recorded yet -->
			{#if state === 'starting'}
				<span class="shrink-0 text-caption text-fg-muted" role="status">
					{VOICE_MODE_MESSAGE.starting}
				</span>
			{:else}
				<span class="w-10 shrink-0 text-right text-caption text-fg-muted tabular-nums">
					{formatClipTime(elapsedMs)}
				</span>
			{/if}
		{/if}

		<ChatFormActionRecord
			class="shrink-0"
			disabled={state !== 'recording'}
			isRecording
			onMicClick={onStop}
		/>

		{#if onSend}
			<Button
				class="size-8 shrink-0 rounded-full p-0 max-md:size-9"
				disabled={state !== 'recording'}
				onclick={onSend}
				type="button"
			>
				<span class="sr-only">Stop and send</span>

				<ArrowUp class="size-icon-md" />
			</Button>
		{/if}
	</div>

	{#if liveCaption}
		<p
			aria-live="polite"
			class="line-clamp-2 px-10 text-body-sm break-words text-fg-muted max-md:px-11"
			data-slot="chat-form-recording-caption"
		>
			<VoiceSpokenText live text={liveCaption} />
		</p>
	{/if}
</div>
