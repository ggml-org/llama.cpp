<script lang="ts" module>
	import { userAttachments } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageUserPending from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserPending.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: ChatMessageUserPending,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatMessages/ChatMessageUserPending'
	});

	const props = {
		content: 'Also check the tests folder before you answer.',
		onDelete: () => {},
		onEdit: () => {},
		onSendImmediately: () => {}
	};
</script>

<Story asChild name="States">
	<StateGallery columns={1}>
		<StateGalleryItem label="Default" note="hover for actions">
			<ChatMessageUserPending {...props} class="w-full" />
		</StateGalleryItem>

		<StateGalleryItem label="With attachments">
			<ChatMessageUserPending {...props} class="w-full" extras={[userAttachments[0]]} />
		</StateGalleryItem>
	</StateGallery>
</Story>
