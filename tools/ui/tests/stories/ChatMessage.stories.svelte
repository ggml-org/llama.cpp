<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import type { ChatMessageActions } from '$lib/types';

	const { Story } = defineMeta({
		component: ChatMessage,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessage'
	});

	const chatActions: ChatMessageActions = {
		continueAssistantMessage: () => {},
		copy: () => {},
		delete: () => {},
		editUserMessagePreserveResponses: () => {},
		editWithBranching: () => {},
		editWithReplacement: () => {},
		forkConversation: () => {},
		navigateToSibling: () => {},
		regenerateWithBranching: () => {}
	};

	// Mock messages for different scenarios
	const userMessage: DatabaseMessage = {
		children: [],
		content: 'What is the meaning of life, the universe, and everything?',
		convId: 'conv-1',
		id: '1',
		parent: '',
		role: 'user',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 5,
		type: 'message'
	};

	const assistantMessage: DatabaseMessage = {
		children: [],
		content:
			'The answer to the ultimate question of life, the universe, and everything is **42**.\n\nThis comes from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy," where a supercomputer named Deep Thought calculated this answer over 7.5 million years. However, the question itself was never properly formulated, which is why the answer seems meaningless without context.',
		convId: 'conv-1',
		id: '2',
		parent: '1',
		role: 'assistant',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 3,
		type: 'message'
	};

	const assistantWithReasoning: DatabaseMessage = {
		children: [],
		content: "Here's the concise answer, now that I've thought it through carefully for you.",
		convId: 'conv-1',
		id: '3',
		parent: '1',
		role: 'assistant',
		thinking:
			"Let's consider the user's question step by step:\\n\\n1. Identify the core problem\\n2. Evaluate relevant information\\n3. Formulate a clear answer\\n\\nFollowing this process ensures the final response stays focused and accurate.",
		timestamp: Date.now() - 1000 * 60 * 2,
		type: 'message'
	};
	const rawOutputMessage: DatabaseMessage = {
		children: [],
		content:
			'<|channel|>analysis<|message|>User greeted me. Initiating overcomplicated analysis: Is this a trap? No, just a normal hello. Respond calmly, act like a helpful assistant, and do not start explaining quantum physics again. Confidence 0.73. Engaging socially acceptable greeting protocol...<|end|>Hello there! How can I help you today?',
		convId: 'conv-1',
		id: '6',
		parent: '1',
		role: 'assistant',
		thinking: '',
		timestamp: Date.now() - 1000 * 60,
		type: 'message'
	};

	let processingMessage = $state({
		children: [],
		content: '',
		convId: 'conv-1',
		id: '4',
		parent: '1',
		role: 'assistant',
		thinking: '',
		timestamp: 0, // No timestamp = processing
		type: 'message'
	});

	let streamingMessage = $state({
		children: [],
		content: '',
		convId: 'conv-1',
		id: '5',
		parent: '1',
		role: 'assistant',
		thinking: '',
		timestamp: 0, // No timestamp = streaming
		type: 'message'
	});
</script>

<Story
	name="User"
	args={{
		message: userMessage
	}}
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	name="Assistant"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantMessage
	}}
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	name="AssistantWithReasoning"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantWithReasoning
	}}
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	name="RawLlmOutput"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: rawOutputMessage
	}}
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', true);
	}}
/>

<Story
	name="WithReasoningContent"
	args={{
		message: streamingMessage
	}}
	asChild
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
		// Phase 1: Stream reasoning content in chunks
		let reasoningText =
			'I need to think about this carefully. Let me break down the problem:\n\n1. The user is asking for help with something complex\n2. I should provide a thorough and helpful response\n3. I need to consider multiple approaches\n4. The best solution would be to explain step by step\n\nThis approach will ensure clarity and understanding.';
		let reasoningChunk = 'I';
		let i = 0;

		while (i < reasoningText.length) {
			const chunkSize = Math.floor(Math.random() * 5) + 3; // Random 3-7 characters
			const chunk = reasoningText.slice(i, i + chunkSize);

			reasoningChunk += chunk;

			// Update the reactive state directly
			streamingMessage.thinking = reasoningChunk;

			i += chunkSize;
			await new Promise((resolve) => setTimeout(resolve, 50));
		}

		const regularText =
			"Based on my analysis, here's the solution:\n\n**Step 1:** First, we need to understand the requirements clearly.\n\n**Step 2:** Then we can implement the solution systematically.\n\n**Step 3:** Finally, we test and validate the results.\n\nThis approach ensures we cover all aspects of the problem effectively.";

		let contentChunk = '';

		i = 0;

		while (i < regularText.length) {
			const chunkSize = Math.floor(Math.random() * 5) + 3; // Random 3-7 characters
			const chunk = regularText.slice(i, i + chunkSize);

			contentChunk += chunk;

			// Update the reactive state directly
			streamingMessage.content = contentChunk;

			i += chunkSize;
			await new Promise((resolve) => setTimeout(resolve, 50));
		}

		streamingMessage.timestamp = Date.now();
	}}
>
	<div class="w-[56rem]">
		<ChatMessage message={streamingMessage} {chatActions} />
	</div>
</Story>

<Story
	name="Processing"
	args={{
		message: processingMessage
	}}
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
		// Import the chat store to simulate loading state
		const { chatStore } = await import('$lib/stores/chat/index.svelte');

		// Set loading state to true to trigger the processing UI
		chatStore.isLoading = true;

		// Simulate the processing state hook behavior
		// This will show the "Generating..." text and parameter details
		await new Promise((resolve) => setTimeout(resolve, 100));
	}}
/>
