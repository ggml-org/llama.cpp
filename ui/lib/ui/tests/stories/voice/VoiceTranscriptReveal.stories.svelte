<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { demoClip } from './fixtures';
	import {
		VoiceTranscriptReveal,
		VoiceTranscriptStatus,
		VoiceWaveform
	} from '@ggml-org/llama-ui/voice';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect, waitFor } from 'storybook/test';

	// a voice message with its transcript: hidden at first, opened under the clip on hover, keyboard focus or a tap
	const { Story } = defineMeta({
		component: VoiceTranscriptReveal,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Voice/VoiceTranscriptReveal'
	});

	const clip = demoClip(6, 5);

	const TRANSCRIPT = 'Can you summarise the three main points from the meeting notes I attached?';
	const LONG_TRANSCRIPT =
		"Okay, so here's what I'm thinking for the trip. We fly into Lisbon on Friday evening, spend the weekend there, and then on Monday take the train up to Porto. I'd like to stay somewhere near the river, nothing fancy, and I want at least one day with no plans at all. Can you put together a rough itinerary with a couple of options for each day, and flag anything we should book ahead, like restaurants or the tram tours?";

	const BUBBLE_CLASS =
		'w-80 max-w-full rounded-2xl matte-sheen bg-(--user-bubble-background) px-2.5 py-2';

	function region(root: HTMLElement): HTMLElement {
		return root.querySelector<HTMLElement>('[data-slot="voice-transcript"]')!;
	}

	function reveal(root: HTMLElement): HTMLElement {
		return root.querySelector<HTMLElement>('[data-slot="voice-transcript-reveal"]')!;
	}
</script>

{#snippet message(
	status?: VoiceTranscriptStatus,
	transcript: string | null = TRANSCRIPT,
	errorReason: string | null = null
)}
	<div class={BUBBLE_CLASS}>
		<VoiceTranscriptReveal {errorReason} {status} {transcript}>
			<VoiceWaveform durationMs={clip.durationMs} peaks={clip.peaks} src={clip.src} />
		</VoiceTranscriptReveal>
	</div>
{/snippet}

<Story asChild name="States">
	<StateGallery minItemWidth="20rem">
		<StateGalleryItem label="Done" note="hover or tab in to read it">
			{@render message()}
		</StateGalleryItem>

		<StateGalleryItem label="Transcribing">
			{@render message(VoiceTranscriptStatus.TRANSCRIBING, null)}
		</StateGalleryItem>

		<StateGalleryItem label="Error">
			{@render message(VoiceTranscriptStatus.ERROR, null)}
		</StateGalleryItem>

		<StateGalleryItem label="Error with its reason" note="when the app can say why">
			{@render message(
				VoiceTranscriptStatus.ERROR,
				null,
				'This model only transcribes live, over a connection a browser cannot open on a server with an API key.'
			)}
		</StateGalleryItem>

		<StateGalleryItem label="None" note="no transcript asked for: nothing opens">
			{@render message(VoiceTranscriptStatus.IDLE, null)}
		</StateGalleryItem>

		<StateGalleryItem label="Long text">
			{@render message(VoiceTranscriptStatus.DONE, LONG_TRANSCRIPT)}
		</StateGalleryItem>

		<StateGalleryItem label="No speech" note="done, but empty">
			{@render message(VoiceTranscriptStatus.DONE, '')}
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- separate: play hovers the clip, then tabs into it, and checks the transcript opens each time -->
<Story
	asChild
	name="HoverAndFocus"
	play={async ({ canvas, canvasElement, userEvent }) => {
		const root = reveal(canvasElement);

		// closed, but in the DOM for screen readers
		// drawn word by word, see VoiceSpokenText
		await expect(region(canvasElement)).toHaveTextContent(TRANSCRIPT);
		await expect(root).not.toHaveAttribute('data-revealed');
		await expect(getComputedStyle(region(canvasElement)).opacity).toBe('0');

		await userEvent.hover(root);
		await waitFor(() => expect(root).toHaveAttribute('data-revealed'));
		await waitFor(() => expect(getComputedStyle(region(canvasElement)).opacity).toBe('1'));

		await userEvent.unhover(root);
		await waitFor(() => expect(root).not.toHaveAttribute('data-revealed'));

		await userEvent.tab();
		await expect(canvas.getByRole('button', { name: 'Play voice message' })).toHaveFocus();
		await waitFor(() => expect(root).toHaveAttribute('data-revealed'));

		await userEvent.tab();
		await expect(canvas.getByRole('slider')).toHaveFocus();
		await expect(root).toHaveAttribute('data-revealed');

		await userEvent.tab();
		await waitFor(() => expect(root).not.toHaveAttribute('data-revealed'));
	}}
>
	<div class="flex flex-col gap-4">
		{@render message()}

		<button class="text-caption text-muted-foreground" type="button">After the message</button>
	</div>
</Story>

<!-- separate: play taps the clip with a finger, then closes it with the toggle key -->
<Story
	asChild
	name="Touch"
	play={async ({ canvas, canvasElement, userEvent }) => {
		const root = reveal(canvasElement);
		const time = canvas.getByText('0:06');

		await userEvent.pointer({ keys: '[TouchA]', target: time });
		await waitFor(() => expect(root).toHaveAttribute('data-revealed'));

		const hide = canvas.getByRole('button', { name: 'Hide transcript' });

		await expect(hide).toHaveAttribute('aria-expanded', 'true');
		await userEvent.pointer({ keys: '[TouchA]', target: hide });
		await waitFor(() => expect(root).not.toHaveAttribute('data-revealed'));
	}}
>
	<StateGallery columns={1} width="24rem">
		<StateGalleryItem
			label="Touch"
			note="a tap outside the controls toggles it; with no hover a toggle key shows"
		>
			{@render message()}
		</StateGalleryItem>
	</StateGallery>
</Story>
