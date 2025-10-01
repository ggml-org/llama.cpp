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
		content:
			'The answer to the ultimate question of life, the universe, and everything is **42**.\n\nThis comes from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy," where a supercomputer named Deep Thought calculated this answer over 7.5 million years. However, the question itself was never properly formulated, which is why the answer seems meaningless without context.',
		parent: '1',
		thinking: '',
		children: []
	};

	let processingMessage = $state({
		id: '4',
		convId: 'conv-1',
		type: 'message',
		timestamp: 0, // No timestamp = processing
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	});

	let streamingMessage = $state({
		id: '5',
		convId: 'conv-1',
		type: 'message',
		timestamp: 0, // No timestamp = streaming
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	});

	// Message with <think> format thinking content
	const thinkTagMessage: DatabaseMessage = {
		id: '6',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 60 * 2,
		role: 'assistant',
		content:
			"<think>\nLet me analyze this step by step:\n\n1. The user is asking about thinking formats\n2. I need to demonstrate the &lt;think&gt; tag format\n3. This content should be displayed in the thinking section\n4. The main response should be separate\n\nThis is a good example of reasoning content.\n</think>\n\nHere's my response after thinking through the problem. The thinking content above should be displayed separately from this main response content.",
		parent: '1',
		thinking: '',
		children: []
	};

	// Message with [THINK] format thinking content
	const thinkBracketMessage: DatabaseMessage = {
		id: '7',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 60 * 1,
		role: 'assistant',
		content:
			'[THINK]\nThis is the DeepSeek-style thinking format:\n\n- Using square brackets instead of angle brackets\n- Should work identically to the &lt;think&gt; format\n- Content parsing should extract this reasoning\n- Display should be the same as &lt;think&gt; format\n\nBoth formats should be supported seamlessly.\n[/THINK]\n\nThis is the main response content that comes after the [THINK] block. The reasoning above should be parsed and displayed in the thinking section.',
		parent: '1',
		thinking: '',
		children: []
	};

	// Message with ◁think▷ format thinking content
	const thinkPipeMessage: DatabaseMessage = {
		id: '8',
		convId: 'conv-1',
		type: 'message',
		timestamp: Date.now() - 1000 * 30,
		role: 'assistant',
		content:
			"◁think▷\nThis demonstrates the pipe-style thinking format:\n\n- Uses &#9665;think&#9655; and &#9665;/think&#9665; tags\n- Common in some model architectures\n- Should parse identically to other formats\n- Provides the same separation of reasoning and response\n\nAll three formats offer equivalent functionality.\n◁/think▷\n\nHere's my response using the &#9665;think&#9655; format. The reasoning content above should be extracted and displayed in the thinking section, while this main content appears in the regular message area.",
		parent: '1',
		thinking: '',
		children: []
	};

	// Streaming message for <think> format
	let streamingThinkMessage = $state({
		id: '9',
		convId: 'conv-1',
		type: 'message',
		timestamp: 0, // No timestamp = streaming
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	});

	// Streaming message for [THINK] format
	let streamingBracketMessage = $state({
		id: '10',
		convId: 'conv-1',
		type: 'message',
		timestamp: 0, // No timestamp = streaming
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	});

	// Streaming message for ◁think▷ format
	let streamingPipeMessage = $state({
		id: '11',
		convId: 'conv-1',
		type: 'message',
		timestamp: 0, // No timestamp = streaming
		role: 'assistant',
		content: '',
		parent: '1',
		thinking: '',
		children: []
	});
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
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: assistantMessage
	}}
/>

<Story
	name="WithThinkingBlock"
	args={{
		message: streamingMessage
	}}
	asChild
	play={async () => {
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
		<ChatMessage message={streamingMessage} />
	</div>
</Story>

<Story
	name="Processing"
	args={{
		message: processingMessage
	}}
	play={async () => {
		// Import the chat store to simulate loading state
		const { chatStore } = await import('$lib/stores/chat.svelte');
		
		// Set loading state to true to trigger the processing UI
		chatStore.isLoading = true;
		
		// Simulate the processing state hook behavior
		// This will show the "Generating..." text and parameter details
		await new Promise(resolve => setTimeout(resolve, 100));
	}}
/>

<Story
	name="ThinkTagFormat"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: thinkTagMessage
	}}
/>

<Story
	name="ThinkBracketFormat"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: thinkBracketMessage
	}}
/>

<Story
	name="ThinkPipeFormat"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		message: thinkPipeMessage
	}}
/>

<Story
	name="StreamingThinkTag"
	args={{
		message: streamingThinkMessage
	}}
	parameters={{
		test: {
			timeout: 30000
		}
	}}
	asChild
	play={async () => {
		// Phase 1: Stream <think> reasoning content
		const thinkingContent =
			'Let me work through this problem systematically:\n\n1. First, I need to understand what the user is asking\n2. Then I should consider different approaches\n3. I need to evaluate the pros and cons\n4. Finally, I should provide a clear recommendation\n\nThis step-by-step approach will ensure accuracy.';

		let currentContent = '<think>\n';
		streamingThinkMessage.content = currentContent;

		for (let i = 0; i < thinkingContent.length; i++) {
			currentContent += thinkingContent[i];
			streamingThinkMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 5));
		}

		// Close the thinking block
		currentContent += '\n</think>\n\n';
		streamingThinkMessage.content = currentContent;
		await new Promise((resolve) => setTimeout(resolve, 200));

		// Phase 2: Stream main response content
		const responseContent =
			"Based on my analysis above, here's the solution:\n\n**Key Points:**\n- The approach should be systematic\n- We need to consider all factors\n- Implementation should be step-by-step\n\nThis ensures the best possible outcome.";

		for (let i = 0; i < responseContent.length; i++) {
			currentContent += responseContent[i];
			streamingThinkMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 10));
		}

		streamingThinkMessage.timestamp = Date.now();
	}}
>
	<div class="w-[56rem]">
		<ChatMessage message={streamingThinkMessage} />
	</div>
</Story>

<Story
	name="StreamingThinkBracket"
	args={{
		message: streamingBracketMessage
	}}
	parameters={{
		test: {
			timeout: 30000
		}
	}}
	asChild
	play={async () => {
		// Phase 1: Stream [THINK] reasoning content
		const thinkingContent =
			'Using the DeepSeek format now:\n\n- This demonstrates the &#91;THINK&#93; bracket format\n- Should parse identically to &lt;think&gt; tags\n- The UI should display this in the thinking section\n- Main content should be separate\n\nAll three formats provide the same functionality.';

		let currentContent = '[THINK]\n';
		streamingBracketMessage.content = currentContent;

		for (let i = 0; i < thinkingContent.length; i++) {
			currentContent += thinkingContent[i];
			streamingBracketMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 5));
		}

		// Close the thinking block
		currentContent += '\n[/THINK]\n\n';
		streamingBracketMessage.content = currentContent;
		await new Promise((resolve) => setTimeout(resolve, 200));

		// Phase 2: Stream main response content
		const responseContent =
			"Here's my response after using the &#91;THINK&#93; format:\n\n**Observations:**\n- All three formats (&lt;think&gt;, &#91;THINK&#93;, &lt;|think|&gt;) work seamlessly\n- The parsing logic handles all cases\n- UI display is consistent across formats\n\nThis demonstrates comprehensive thinking content support.";

		for (let i = 0; i < responseContent.length; i++) {
			currentContent += responseContent[i];
			streamingBracketMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 10));
		}

		streamingBracketMessage.timestamp = Date.now();
	}}
>
	<div class="w-[56rem]">
		<ChatMessage message={streamingBracketMessage} />
	</div>
</Story>

<Story
	name="StreamingThinkPipe"
	args={{
		message: streamingPipeMessage
	}}
	parameters={{
		test: {
			timeout: 30000
		}
	}}
	asChild
	play={async () => {
		// Phase 1: Stream ◁think▷ reasoning content
		const thinkingContent =
			'Demonstrating the pipe format thinking:\n\n- This uses the &#9665;think&#9655; and &#9665;/think&#9655; tag structure\n- Should integrate seamlessly with existing parsing logic\n- UI display should be identical to other formats\n- All three formats provide equivalent functionality\n\nThe pipe format adds another option for reasoning content.';

		let currentContent = '◁think▷\n';
		streamingPipeMessage.content = currentContent;

		for (let i = 0; i < thinkingContent.length; i++) {
			currentContent += thinkingContent[i];
			streamingPipeMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 5));
		}

		// Close the thinking block
		currentContent += '\n◁/think▷\n\n';
		streamingPipeMessage.content = currentContent;
		await new Promise((resolve) => setTimeout(resolve, 200));

		// Phase 2: Stream main response content
		const responseContent =
			"Here's the response after using &#9665;think&#9655; format:\n\n**Key Features:**\n- All three thinking formats (&lt;think&gt;, &#91;THINK&#93;, &lt;|think|&gt;) work identically\n- Parsing logic handles each format seamlessly\n- UI presentation is consistent across all formats\n\nThis demonstrates comprehensive thinking content support.";

		for (let i = 0; i < responseContent.length; i++) {
			currentContent += responseContent[i];
			streamingPipeMessage.content = currentContent;
			await new Promise((resolve) => setTimeout(resolve, 10));
		}

		streamingPipeMessage.timestamp = Date.now();
	}}
>
	<div class="w-[56rem]">
		<ChatMessage message={streamingPipeMessage} />
	</div>
</Story>
