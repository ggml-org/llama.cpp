<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { demoClip } from './fixtures';
	import { VoiceWaveform, VoiceWaveformSize } from '@ggml-org/llama-ui/voice';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect } from 'storybook/test';

	// a recorded voice clip: play, seek along the waveform, the time beside it; the surface is the caller's
	const { Story } = defineMeta({
		component: VoiceWaveform,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Voice/VoiceWaveform'
	});

	const short = demoClip(4.2, 1);
	const long = demoClip(135, 2);
	const other = demoClip(7, 3);

	const CHIP_CLASS = 'w-64 max-w-full rounded-lg border border-border bg-muted py-1.5 pr-3 pl-1.5';
	const BUBBLE_CLASS =
		'w-80 max-w-full rounded-2xl matte-sheen bg-(--user-bubble-background) px-2.5 py-2';
</script>

<Story asChild name="Sizes">
	<StateGallery minItemWidth="18rem">
		<StateGalleryItem label="sm" note="a chip in the chat form">
			<div class={CHIP_CLASS}>
				<VoiceWaveform
					durationMs={short.durationMs}
					peaks={short.peaks}
					size={VoiceWaveformSize.SM}
					src={short.src}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="md" note="a message bubble">
			<div class={BUBBLE_CLASS}>
				<VoiceWaveform durationMs={short.durationMs} peaks={short.peaks} src={short.src} />
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="States">
	<StateGallery columns={1} width="28rem">
		<StateGalleryItem label="Short clip" note="0:04">
			<div class={BUBBLE_CLASS}>
				<VoiceWaveform durationMs={short.durationMs} peaks={short.peaks} src={short.src} />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Long clip" note="2:15">
			<div class={BUBBLE_CLASS}>
				<VoiceWaveform durationMs={long.durationMs} peaks={long.peaks} src={long.src} />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="No audio" note="no src: the waveform and length only">
			<div class={BUBBLE_CLASS}>
				<VoiceWaveform durationMs={short.durationMs} peaks={short.peaks} />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="No peaks yet" note="a flat line">
			<div class={CHIP_CLASS}>
				<VoiceWaveform durationMs={short.durationMs} peaks={[]} size={VoiceWaveformSize.SM} src={short.src} />
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- separate: play seeks with the keys and a press on the bars; pressing play needs a real click (tests/client/voice-waveform.svelte.test.ts) -->
<Story
	asChild
	name="Playback"
	play={async ({ canvas, userEvent }) => {
		await expect(canvas.getByRole('button', { name: 'Play first clip' })).toBeEnabled();
		await expect(canvas.getByRole('button', { name: 'Play second clip' })).toBeEnabled();

		const slider = canvas.getByRole('slider', { name: 'Long clip position' });

		slider.focus();
		await userEvent.keyboard('{End}');
		await expect(slider).toHaveAttribute('aria-valuenow', '135');
		await userEvent.keyboard('{ArrowLeft}');
		await expect(slider).toHaveAttribute('aria-valuenow', '130');
		await expect(slider).toHaveAttribute('aria-valuetext', '2:10 of 2:15');
		await userEvent.keyboard('{Home}');
		await userEvent.keyboard('{ArrowRight}');
		await expect(slider).toHaveAttribute('aria-valuenow', '5');

		// a press halfway along the bars seeks to the middle
		const rect = slider.getBoundingClientRect();

		await userEvent.pointer({
			coords: { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 },
			keys: '[MouseLeft]',
			target: slider
		});
		await expect(Number(slider.getAttribute('aria-valuenow'))).toBeCloseTo(67.5, -1);
	}}
>
	<div class="flex w-80 flex-col gap-3">
		<div class={BUBBLE_CLASS}>
			<VoiceWaveform
				durationMs={short.durationMs}
				label="First clip"
				peaks={short.peaks}
				src={short.src}
			/>
		</div>

		<div class={BUBBLE_CLASS}>
			<VoiceWaveform
				durationMs={other.durationMs}
				label="Second clip"
				peaks={other.peaks}
				src={other.src}
			/>
		</div>

		<div class={BUBBLE_CLASS}>
			<VoiceWaveform
				durationMs={long.durationMs}
				label="Long clip"
				peaks={long.peaks}
				src={long.src}
			/>
		</div>
	</div>
</Story>

<!-- separate: play checks that a clip without audio offers nothing to press -->
<Story
	asChild
	name="NoSource"
	play={async ({ canvas }) => {
		await expect(canvas.getByRole('button', { name: 'Play voice message' })).toBeDisabled();
		await expect(canvas.getByRole('slider')).toHaveAttribute('aria-disabled', 'true');
		await expect(canvas.getByText('0:04')).toBeInTheDocument();
	}}
>
	<div class={BUBBLE_CLASS}>
		<VoiceWaveform durationMs={short.durationMs} peaks={short.peaks} />
	</div>
</Story>
