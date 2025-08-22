<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage.svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatScreen/ChatMessage',
		component: ChatMessage,
		parameters: {
			layout: 'centered'
		}
	});

	// Mock messages for different scenarios
	const userMessage: DatabaseMessage = {
		id: '1',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 60 * 5,
		role: 'user',
		content: 'What is the meaning of life, the universe, and everything?',
		parent: '',
		thinking: '',
		children: []
	};

	const assistantMessage: DatabaseMessage = {
		id: '2',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 60 * 3,
		role: 'assistant',
		content: 'The answer to the ultimate question of life, the universe, and everything is **42**.\n\nThis comes from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy," where a supercomputer named Deep Thought calculated this answer over 7.5 million years. However, the question itself was never properly formulated, which is why the answer seems meaningless without context.',
		parent: '1',
		thinking: '',
		children: []
	};

	const thinkingMessage: DatabaseMessage = {
		id: '3',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 60 * 2,
		role: 'assistant',
		content: 'Let me solve this step by step.\n\nFirst, I need to understand what you\'re asking for. Then I\'ll work through the problem systematically.',
		parent: '1',
		thinking: 'The user is asking me to solve a complex problem. I should break this down into steps:\n\n1. Understand the requirements\n2. Analyze the problem\n3. Consider different approaches\n4. Choose the best solution\n5. Implement and explain\n\nThis seems like a good approach to take.',
		children: []
	};

	const processingMessage: DatabaseMessage = {
		id: '4',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now(),
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	};
</script>

<Story
	name="User"
	args={{
		message: userMessage
	}}
/>

<Story
	name="Assistant"
	args={{
		message: assistantMessage
	}}
/>

<Story
	name="ThinkingBlock"
	args={{
		message: thinkingMessage
	}}
/>

<Story
	name="ProcessingState"
	args={{
		message: processingMessage
	}}
	play={({ canvasElement }) => {
		// Simulate processing state by setting up mock processing data
		const processingState = {
			slots: {
				'slot-1': { content: 'Processing your request...', timestamp: Date.now() },
				'slot-2': { content: 'Analyzing data...', timestamp: Date.now() + 1000 }
			}
		};
		
		// This would normally be handled by the useProcessingState hook
		// but for Storybook we can simulate the visual state
	}}
/>
