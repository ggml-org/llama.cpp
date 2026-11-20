<script lang="ts" module>
	import { demoLevel } from './fixtures';
	import VoiceLiveDemo from './VoiceLiveDemo.svelte';
	import { VoiceWaveformLive } from '@ggml-org/llama-ui/voice';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';
	import { expect, waitFor } from 'storybook/test';

	// the waveform while recording, in the chat form's input row; here a fake microphone meter feeds it
	const { Story } = defineMeta({
		component: VoiceWaveformLive,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Voice/VoiceWaveformLive'
	});

	const frozen = Array.from({ length: 48 }, (_, index) => demoLevel(index / 20, 4));

	function barHeights(root: HTMLElement): string {
		return [...root.querySelectorAll<HTMLElement>('[data-slot="voice-waveform-live"] span[style]')]
			.map((bar) => bar.style.height)
			.join(',');
	}
</script>

<Story
	asChild
	name="Recording"
	play={async ({ canvasElement }) => {
		const live = canvasElement.querySelector('[data-slot="voice-waveform-live"]')!;

		await expect(live.querySelector('.sr-only')).toHaveTextContent('Recording');
		await expect(live.querySelector('[aria-hidden="true"]')).toBeInTheDocument();

		const before = barHeights(canvasElement);

		await waitFor(() => expect(barHeights(canvasElement)).not.toBe(before));
	}}
>
	<StateGallery columns={1} width="40rem">
		<StateGalleryItem label="Recording" note="a level every 50 ms, newest on the right">
			<VoiceLiveDemo />
		</StateGalleryItem>

		<StateGalleryItem label="Paused" note="active off: the bars turn muted and stop">
			<VoiceLiveDemo active={false} initialSeconds={6} seed={2} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Levels">
	<StateGallery columns={1} width="32rem">
		<StateGalleryItem label="Filling up" note="fewer levels than bars: resting dots on the left">
			<VoiceWaveformLive levels={frozen.slice(0, 12)} />
		</StateGalleryItem>

		<StateGalleryItem label="Full" note="the last levels that fit">
			<VoiceWaveformLive levels={frozen} />
		</StateGalleryItem>

		<StateGalleryItem label="Silence">
			<VoiceWaveformLive levels={[]} />
		</StateGalleryItem>

		<StateGalleryItem label="Inactive">
			<VoiceWaveformLive active={false} levels={frozen} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story
	asChild
	globals={{ viewport: { isRotated: false, value: 'mobile2' } }}
	name="Narrow"
	parameters={{ layout: 'fullscreen' }}
>
	<div class="p-4">
		<VoiceLiveDemo />
	</div>
</Story>
