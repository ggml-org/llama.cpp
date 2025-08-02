<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatMessage } from '$lib/components';

	const { Story } = defineMeta({
		title: 'Components/ChatMessage',
		component: ChatMessage,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story
	name="UserMessage"
	args={{
		message: {
			role: 'user',
			content: 'Hello! Can you help me understand how to use this chat interface?'
		}
	}}
/>

<Story
	name="AssistantMessage"
	args={{
		message: {
			role: 'assistant',
			content: 'Hello'
		}
	}}
/>
