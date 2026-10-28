<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormActionRecord from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';

	const { Story } = defineMeta({
		component: ChatFormActionRecord,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormActionRecord'
	});
</script>

<Story asChild name="States">
	<StateGallery columns={5}>
		<StateGalleryItem label="Idle">
			<ChatFormActionRecord hasAudioModality />
		</StateGalleryItem>

		<StateGalleryItem label="Recording">
			<ChatFormActionRecord hasAudioModality isRecording />
		</StateGalleryItem>

		<StateGalleryItem label="Loading">
			<ChatFormActionRecord hasAudioModality isLoading />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ChatFormActionRecord disabled hasAudioModality />
		</StateGalleryItem>

		<StateGalleryItem label="No audio" note="model lacks audio">
			<ChatFormActionRecord hasAudioModality={false} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- separate: play hovers the button to open its tooltip -->
<Story
	args={{ hasAudioModality: false }}
	name="NoAudioModalityTooltip"
	play={async ({ canvas, userEvent }) => {
		// the tooltip trigger wraps the disabled button, hover the trigger
		await userEvent.hover(canvas.getAllByRole('button', { name: 'Start recording' })[0]);
	}}
/>
