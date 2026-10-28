<script lang="ts" module>
	import { STORY_CLASS, userAttachments } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageUserPending from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserPending.svelte';

	const { Story } = defineMeta({
		args: {
			class: STORY_CLASS,
			content: 'Also check the tests folder before you answer.',
			onDelete: () => {},
			onEdit: () => {},
			onSendImmediately: () => {}
		},
		component: ChatMessageUserPending,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageUserPending'
	});
</script>

<Story name="Default" />

<Story args={{ extras: [userAttachments[0]] }} name="WithAttachments" />
