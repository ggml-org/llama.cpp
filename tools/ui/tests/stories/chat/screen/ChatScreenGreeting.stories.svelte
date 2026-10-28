<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenGreeting from '$lib/components/app/chat/ChatScreen/ChatScreenGreeting.svelte';
	import { serverStore } from '$lib/stores';

	const { Story } = defineMeta({
		component: ChatScreenGreeting,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatScreenGreeting'
	});

	function seedAudio(audio: boolean) {
		serverStore.props = {
			...serverStore.props,
			modalities: { audio, video: false, vision: false }
		} as ApiLlamaCppServerProps;
	}
</script>

<!-- the wrapper stands in for the form wrapper: on desktop the greeting sits in its top padding -->
<Story asChild beforeEach={() => seedAudio(false)} name="Default">
	<div class="relative h-40 w-[40rem]" style="--chat-form-padding-top: 10rem">
		<ChatScreenGreeting isEmpty />
	</div>
</Story>

<Story asChild beforeEach={() => seedAudio(true)} name="WithAudioModality">
	<div class="relative h-40 w-[40rem]" style="--chat-form-padding-top: 10rem">
		<ChatScreenGreeting isEmpty />
	</div>
</Story>
