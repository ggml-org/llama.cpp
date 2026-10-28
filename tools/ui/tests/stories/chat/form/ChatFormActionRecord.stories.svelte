<script lang="ts" module>
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

<Story args={{ hasAudioModality: true }} name="Default" />

<Story args={{ hasAudioModality: true, isRecording: true }} name="Recording" />

<Story args={{ hasAudioModality: true, isLoading: true }} name="Loading" />

<Story args={{ disabled: true, hasAudioModality: true }} name="Disabled" />

<Story
	args={{ hasAudioModality: false }}
	name="NoAudioModality"
	play={async ({ canvas, userEvent }) => {
		// the tooltip trigger wraps the disabled button, hover the trigger
		await userEvent.hover(canvas.getAllByRole('button', { name: 'Start recording' })[0]);
	}}
/>
