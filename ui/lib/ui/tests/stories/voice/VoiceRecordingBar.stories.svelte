<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { VoiceRecordingBar, VoiceRecordingBarStatus } from '@ggml-org/llama-ui/voice';
	import { Square } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect, fn } from 'storybook/test';

	// the row while recording: cancel, the waveform and time, the app's stop button, send; text and warnings under it
	const { Story } = defineMeta({
		component: VoiceRecordingBar,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Voice/VoiceRecordingBar'
	});

	/** Syllable-like levels, newest last. */
	const levels = Array.from({ length: 48 }, (_, index) =>
		Math.max(0.08, Math.abs(Math.sin(index / 2.2)) * (0.4 + 0.3 * Math.sin(index / 7)))
	);

	const ROW_CLASS = 'w-[28rem] max-w-full rounded-2xl border border-border bg-muted p-2';
</script>

{#snippet stopButton({ disabled }: { disabled: boolean })}
	<button
		class="flex size-8 shrink-0 items-center justify-center rounded-full bg-destructive text-white disabled:opacity-50"
		{disabled}
		type="button"
	>
		<span class="sr-only">Stop recording</span>

		<Square class="size-3.5 fill-current" />
	</button>
{/snippet}

<Story asChild name="States">
	<StateGallery columns={1} width="30rem">
		<StateGalleryItem label="Starting" note="no audio yet: stop and send wait">
			<div class={ROW_CLASS}>
				<VoiceRecordingBar
					elapsedMs={0}
					levels={[]}
					onCancel={fn()}
					onSend={fn()}
					status={VoiceRecordingBarStatus.STARTING}
					stop={stopButton}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Recording" note="the time runs from the first audio">
			<div class={ROW_CLASS}>
				<VoiceRecordingBar
					elapsedMs={12_400}
					label="Recording"
					{levels}
					onCancel={fn()}
					onSend={fn()}
					status={VoiceRecordingBarStatus.RECORDING}
					stop={stopButton}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Caption" note="text streaming in while speaking">
			<div class={ROW_CLASS}>
				<VoiceRecordingBar
					caption="Could you summarise the notes from this morning and list what is still open"
					elapsedMs={6_100}
					{levels}
					onCancel={fn()}
					onSend={fn()}
					status={VoiceRecordingBarStatus.RECORDING}
					stop={stopButton}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Warning" note="e.g. speech too quiet even boosted">
			<div class={ROW_CLASS}>
				<VoiceRecordingBar
					elapsedMs={3_000}
					levels={levels.map((level) => level / 6)}
					onCancel={fn()}
					status={VoiceRecordingBarStatus.RECORDING}
					stop={stopButton}
					warning="Your microphone sounds quiet. Move closer, or turn its input level up."
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Waiting" note="the text is on its way">
			<div class={ROW_CLASS}>
				<VoiceRecordingBar
					elapsedMs={9_000}
					{levels}
					onCancel={fn()}
					status={VoiceRecordingBarStatus.WAITING}
					stop={stopButton}
				/>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- separate: play checks that stop and send wait for the first audio, and that the labels can be the app's -->
<Story
	name="Starting, with labels"
	args={{
		elapsedMs: 0,
		labels: { cancel: 'Discard', send: 'Send now', starting: 'Waking the microphone...' },
		levels: [],
		onCancel: fn(),
		onSend: fn(),
		status: VoiceRecordingBarStatus.STARTING,
		stop: stopButton
	}}
	play={async ({ args, canvas, userEvent }) => {
		await expect(canvas.getByText('Waking the microphone...')).toBeVisible();
		await expect(canvas.getByRole('button', { name: 'Send now' })).toBeDisabled();
		await expect(canvas.getByRole('button', { name: 'Stop recording' })).toBeDisabled();

		await userEvent.click(canvas.getByRole('button', { name: 'Discard' }));
		await expect(args.onCancel).toHaveBeenCalledOnce();
	}}
/>
