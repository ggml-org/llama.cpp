<script lang="ts" module>
	import { JOKES_TABLE_MD, WIDE_CELL_TABLE_MD } from './fixtures/mobile-tables.js';
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
		regenerateCompaction: () => {},
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

	const assistantWithTable: DatabaseMessage = {
		children: [],
		content: `${JOKES_TABLE_MD}\n\n${WIDE_CELL_TABLE_MD}`,
		convId: 'conv-1',
		id: '7',
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

	const compactionMessage: DatabaseMessage = {
		children: [],
		content:
			'The user asked about the ultimate answer and its origin. Key facts: the answer is **42**, from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy", computed by Deep Thought over 7.5 million years.\n\nNothing is left to do unless the user wants the question itself discussed.',
		convId: 'conv-1',
		id: '8',
		// The model the server reports for the summary request.
		model: 'ggml-org/Qwen3.8-Flash-Next',
		parent: '1',
		role: 'user',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 30,
		// The pre-encode stamps the reduced prompt on the node; `compaction`
		// keeps the summary request's own timings for the stats row.
		timings: {
			cache_n: 0,
			compaction: {
				cache_n: 0,
				predicted_ms: 12000,
				predicted_n: 300,
				prompt_ms: 18200,
				prompt_n: 42200
			},
			predicted_n: 0,
			prompt_n: 3204
		},
		type: 'compaction'
	};

	// The turn the compaction summarized: its timings are the context size
	// the node's toggle reports as the "before" number.
	const preCompactionAssistant: DatabaseMessage = {
		children: [],
		content: 'Deep Thought took 7.5 million years to answer.',
		convId: 'conv-1',
		id: '7',
		parent: '1',
		role: 'assistant',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 31,
		timings: { cache_n: 0, predicted_n: 118, prompt_n: 42200 },
		type: 'message'
	};

	const compactionPendingMessage: DatabaseMessage = {
		children: [],
		content: '',
		convId: 'conv-1',
		id: '9',
		parent: '1',
		role: 'user',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 29,
		type: 'compaction'
	};

	// The same node once the prompt is through and the summary is streaming.
	const compactionGeneratingMessage: DatabaseMessage = {
		children: [],
		content:
			'The user asked about the ultimate answer and its origin. Key facts: the answer is **42**',
		convId: 'conv-1',
		id: '10',
		model: 'ggml-org/Qwen3.8-Flash-Next',
		parent: '1',
		role: 'user',
		thinking: '',
		timestamp: Date.now() - 1000 * 60 * 29,
		type: 'compaction'
	};

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
	args={{
		message: userMessage
	}}
	name="User"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantMessage
	}}
	name="Assistant"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantWithTable
	}}
	name="AssistantWithTable"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantWithReasoning
	}}
	name="AssistantWithReasoning"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: rawOutputMessage
	}}
	name="RawLlmOutput"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', true);
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: compactionMessage
	}}
	name="Compaction"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');
		const { conversationsStore } = await import('$lib/stores/conversations/index.svelte');

		settingsStore.updateConfig('showMessageStats', true);
		settingsStore.updateConfig('showRawOutputSwitch', false);
		// The toggle reads the before size from the branch, so the story needs
		// the summarized turn in the active message list.
		conversationsStore.activeMessages = [preCompactionAssistant, compactionMessage];
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: compactionPendingMessage
	}}
	name="CompactionPrefill"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');
		const { chatActivityStore } = await import('$lib/stores/chat/activity.svelte');
		const { chatStore } = await import('$lib/stores/chat/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);

		// The summary request reports its prompt progress through the shared
		// processing state; seed one so the story shows the prefill detail.
		chatActivityStore.markCompacting('conv-1', compactionPendingMessage.id);
		chatStore.processing.setActiveConversation('conv-1');
		chatStore.processing.setState('conv-1', {
			contextTotal: 32768,
			contextUsed: 10000,
			hasNextToken: false,
			outputTokensMax: -1,
			outputTokensUsed: 0,
			progressPercent: 42,
			promptProgress: { cache: 0, processed: 4200, time_ms: 6000, total: 10000 },
			promptTokens: 0,
			speculative: false,
			status: 'preparing',
			temperature: 0.8,
			tokensDecoded: 0,
			tokensPerSecond: 0,
			tokensRemaining: -1,
			topP: 0.95
		});
	}}
/>

<Story
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: compactionGeneratingMessage
	}}
	name="CompactionGenerating"
	play={async () => {
		const { settingsStore } = await import('$lib/stores/settings/index.svelte');
		const { chatActivityStore } = await import('$lib/stores/chat/activity.svelte');
		const { chatStore } = await import('$lib/stores/chat/index.svelte');

		settingsStore.updateConfig('showRawOutputSwitch', false);

		// Prompt processing is over, so only the generated tokens are reported.
		chatActivityStore.markCompacting('conv-1', compactionGeneratingMessage.id);
		chatStore.processing.setActiveConversation('conv-1');
		chatStore.processing.setState('conv-1', {
			contextTotal: 32768,
			contextUsed: 45200,
			hasNextToken: true,
			outputTokensMax: -1,
			outputTokensUsed: 128,
			promptTokens: 42200,
			speculative: false,
			status: 'generating',
			temperature: 0.8,
			tokensDecoded: 128,
			tokensPerSecond: 25,
			tokensRemaining: -1,
			topP: 0.95
		});
	}}
/>

<Story
	args={{
		message: streamingMessage
	}}
	asChild
	name="WithReasoningContent"
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
		<ChatMessage {chatActions} message={streamingMessage} />
	</div>
</Story>

<Story
	args={{
		message: processingMessage
	}}
	name="Processing"
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
