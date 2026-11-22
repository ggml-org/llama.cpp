<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormActionRecord from '#lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';
	import { VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';
	import { expect, screen } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatFormActionRecord,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatForm/ChatFormActionRecord'
	});
</script>

<Story asChild name="States">
	<StateGallery columns={6}>
		<StateGalleryItem label="Voice message">
			<ChatFormActionRecord action={VOICE_MODE_MESSAGE.voiceMessage} />
		</StateGalleryItem>

		<StateGalleryItem label="Dictate">
			<ChatFormActionRecord action={VOICE_MODE_MESSAGE.dictate} />
		</StateGalleryItem>

		<StateGalleryItem label="Recording" note="the stop button">
			<ChatFormActionRecord isRecording />
		</StateGalleryItem>

		<StateGalleryItem label="Loading">
			<ChatFormActionRecord isLoading />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ChatFormActionRecord disabled />
		</StateGalleryItem>

		<StateGalleryItem label="Unavailable" note="no model with audio">
			<ChatFormActionRecord unavailableReason={VOICE_MODE_MESSAGE.unavailable} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- separate: play hovers the button to open its tooltip -->
<Story
	args={{ unavailableReason: VOICE_MODE_MESSAGE.unavailable }}
	name="UnavailableTooltip"
	play={async ({ canvas, userEvent }) => {
		const button = canvas.getAllByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage });

		await expect(button.at(-1)).toBeDisabled();
		// the tooltip trigger wraps the disabled button, hover the trigger
		await userEvent.hover(button[0]);
		await expect(
			(await screen.findAllByText(VOICE_MODE_MESSAGE.unavailable)).length
		).toBeGreaterThan(0);
	}}
/>

<Story
	args={{
		action: VOICE_MODE_MESSAGE.dictate,
		hint: VOICE_MODE_MESSAGE.separateModelHint
	}}
	name="DictateWithAnotherModelTooltip"
	play={async ({ canvas, userEvent }) => {
		await userEvent.hover(canvas.getAllByRole('button', { name: VOICE_MODE_MESSAGE.dictate })[0]);
		await expect((await screen.findAllByText(/alongside the chat model/)).length).toBeGreaterThan(
			0
		);
	}}
/>
