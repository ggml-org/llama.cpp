<!--
	The row shown while a recording is under way: cancel, the live waveform and the time, the app's stop button, and
	an optional send, which stops and sends in one go. While the recording is `waiting` (its text is still on the way)
	a spinner takes the waveform's place. A `warning` (e.g. the speech is too quiet) stays inside the row, so nothing
	around it moves: the waveform turns the warning colour, an icon sits by the time, and the text opens above the row
	in a tooltip for a few seconds, then again on hover or focus. Under the row: the `caption`, text that streams in
	while someone speaks.
	It takes everything in and reports cancel and send; recording, and what stopping does, stay in the app, which
	passes its stop button as the `stop` snippet. The default labels are English; an app passes its own through `labels`.
-->
<script lang="ts" module>
	export interface VoiceRecordingBarLabels {
		/** The cancel button, for screen readers */
		cancel: string;
		/** The send button, for screen readers */
		send: string;
		/** In place of the time until the first audio arrives */
		starting: string;
		/** In place of the waveform while the text is on its way */
		waiting: string;
	}
</script>

<script lang="ts">
	import { ArrowUp, LoaderCircle, TriangleAlert, X } from '@lucide/svelte';
	import type { Snippet } from 'svelte';
	import { ICON_CLASS_DEFAULT } from '../../constants/css-classes.constants';
	import {
		VOICE_RECORDING_BAR_LABELS,
		VOICE_WARNING_SHOWN_MS
	} from '../../constants/voice.constants';
	import { VoiceRecordingBarStatus } from '../../enums/voice.enums';
	import { formatClipTime } from '../../voice/waveform';
	import { Button } from '../ui/button';
	import * as Tooltip from '../ui/tooltip';
	import { cn } from '../ui/utils';
	import VoiceSpokenText from './VoiceSpokenText.svelte';
	import VoiceWaveformLive from './VoiceWaveformLive.svelte';

	interface Props {
		/** Input levels, 0..1, newest last. */
		levels: number[];
		/** Time recorded so far. */
		elapsedMs: number;
		status: VoiceRecordingBarStatus;
		/** The stop button; `disabled` while nothing is being recorded yet. */
		stop: Snippet<[{ disabled: boolean }]>;
		onCancel: () => void;
		/** Stops and sends; left out, there is no send button. */
		onSend?: () => void;
		/** What is being recorded, read by screen readers in place of the waveform, e.g. `Dictating`. */
		label?: string;
		/** Text streaming in while someone speaks. */
		caption?: string;
		/** A problem with the recording, said under the row while it lasts. */
		warning?: string | null;
		labels?: Partial<VoiceRecordingBarLabels>;
		class?: string;
	}

	let {
		caption = '',
		class: className,
		elapsedMs,
		label,
		labels = {},
		levels,
		onCancel,
		onSend,
		status,
		stop,
		warning = null
	}: Props = $props();

	let text = $derived({ ...VOICE_RECORDING_BAR_LABELS, ...labels });
	let recording = $derived(status === VoiceRecordingBarStatus.RECORDING);
	let warningOpen = $state(false);

	// a warning that appears opens by itself, as nobody hovers an icon they have not seen yet
	$effect(() => {
		if (!warning) {
			warningOpen = false;

			return;
		}

		warningOpen = true;

		const timer = setTimeout(() => (warningOpen = false), VOICE_WARNING_SHOWN_MS);

		return () => clearTimeout(timer);
	});
</script>

<div class={cn('flex w-full min-w-0 flex-col gap-1', className)} data-slot="voice-recording-bar">
	<div class="flex w-full min-w-0 items-center gap-2">
		<Button
			class="size-8 shrink-0 rounded-full p-0 text-muted-foreground max-md:size-9"
			onclick={onCancel}
			type="button"
			variant="ghost"
		>
			<span class="sr-only">{text.cancel}</span>

			<X class={ICON_CLASS_DEFAULT} />
		</Button>

		{#if status === VoiceRecordingBarStatus.WAITING}
			<div
				class="flex min-w-0 flex-1 items-center gap-2 px-1 text-caption text-muted-foreground"
				role="status"
			>
				<LoaderCircle class="size-icon-sm animate-spin motion-reduce:animate-none" />
				{text.waiting}
			</div>
		{:else}
			<VoiceWaveformLive
				active={recording}
				class={cn('flex-1 px-1', warning && 'text-fg-warning')}
				{label}
				{levels}
			/>

			{#if warning}
				<Tooltip.Root bind:open={warningOpen}>
					<Tooltip.Trigger>
						{#snippet child({ props })}
							<button
								{...props}
								aria-label={warning}
								class="flex size-5 shrink-0 items-center justify-center rounded-full text-fg-warning"
								data-slot="voice-recording-bar-warning"
								type="button"
							>
								<TriangleAlert aria-hidden="true" class="size-icon-sm" />
							</button>
						{/snippet}
					</Tooltip.Trigger>

					<Tooltip.Content class="max-w-64" side="top">
						<p>{warning}</p>
					</Tooltip.Content>
				</Tooltip.Root>

				<!-- read out when it appears; the tooltip is for the eye -->
				<span class="sr-only" role="status">{warning}</span>
			{/if}

			<!-- the timer starts with the first audio: until then, what is said is not recorded yet -->
			{#if status === VoiceRecordingBarStatus.STARTING}
				<span class="shrink-0 text-caption text-muted-foreground" role="status">{text.starting}</span>
			{:else}
				<span class="w-10 shrink-0 text-right text-caption text-muted-foreground tabular-nums">
					{formatClipTime(elapsedMs)}
				</span>
			{/if}
		{/if}

		{@render stop({ disabled: !recording })}

		{#if onSend}
			<Button
				class="size-8 shrink-0 rounded-full p-0 max-md:size-9"
				disabled={!recording}
				onclick={onSend}
				type="button"
			>
				<span class="sr-only">{text.send}</span>

				<ArrowUp class="size-icon-md" />
			</Button>
		{/if}
	</div>


	{#if caption}
		<p
			aria-live="polite"
			class="line-clamp-2 px-10 text-body-sm break-words text-muted-foreground max-md:px-11"
			data-slot="voice-recording-bar-caption"
		>
			<VoiceSpokenText live text={caption} />
		</p>
	{/if}
</div>
