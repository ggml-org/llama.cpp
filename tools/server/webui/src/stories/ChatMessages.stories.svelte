<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatMessages } from '$lib/components';
	import type { ChatMessageData } from '$lib/types/chat';

	const { Story } = defineMeta({
		title: 'Components/ChatMessages',
		component: ChatMessages,
		parameters: {
			layout: 'centered'
		}
	});

	const sampleMessages: ChatMessageData[] = [
		{
			role: 'user',
			content: 'Hello! Can you help me understand how this chat interface works?',
			timestamp: new Date('2024-01-15T10:00:00Z')
		},
		{
			role: 'assistant',
			content:
				"Of course! I'd be happy to help you understand the chat interface. This is a modern chat application built with SvelteKit that allows you to have conversations with an AI assistant. You can type your messages in the input field at the bottom and press Enter or click the send button to submit them.",
			timestamp: new Date('2024-01-15T10:00:30Z')
		}
	];

	const generalArgs = {
		class: 'w-full max-w-[56rem] m-auto'
	};
</script>

<Story name="Empty" args={{ messages: [], ...generalArgs }} />

<Story name="WithMessages" args={{ messages: sampleMessages, ...generalArgs }} />

<Story name="Loading" args={{ isLoading: true, ...generalArgs }} />
