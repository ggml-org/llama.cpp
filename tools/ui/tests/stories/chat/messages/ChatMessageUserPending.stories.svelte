<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { userAttachments } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageUserPending from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserPending.svelte';

	const { Story } = defineMeta({
		component: ChatMessageUserPending,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageUserPending'
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
