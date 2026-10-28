<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenStreamResumeStatus from '$lib/components/app/chat/ChatScreen/ChatScreenStreamResumeStatus.svelte';
	import { StreamConnectionState } from '$lib/enums';
	import { chatStore } from '$lib/stores';

	const { Story } = defineMeta({
		component: ChatScreenStreamResumeStatus,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatScreenStreamResumeStatus'
	});
</script>

<!-- only RESUMING renders; STREAMING and LOST render nothing -->
<Story
	asChild
	beforeEach={() => {
		chatStore.streamConnectionState = StreamConnectionState.RESUMING;
	}}
	name="Default"
>
	<div class="w-[48rem]">
		<ChatScreenStreamResumeStatus />
	</div>
</Story>
