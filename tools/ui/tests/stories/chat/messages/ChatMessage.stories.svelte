<script lang="ts" module>
	import {
		agenticToolMessages,
		assistantAgenticWithStatsMessage,
		assistantInterruptedReasoningMessage,
		assistantMessage,
		assistantWithReasoningMessage,
		assistantWithStatsMessage,
		assistantWithToolCallsMessage,
		compactionGeneratingMessage,
		compactionMessage,
		compactionPendingMessage,
		CONV_ID,
		cwdChangeMessage,
		cwdClearedMessage,
		generatingProcessingState,
		mcpPromptMessage,
		noopChatActions,
		preCompactionAssistantMessage,
		promptProcessingState,
		rawOutputMessage,
		REASONING_TEXT,
		seedChatStory,
		siblingInfo,
		STORY_CLASS,
		systemLongMessage,
		systemMessage,
		toolResultMessages,
		userAttachmentsOnlyMessage,
		userMarkdownMessage,
		userMessage,
		userWithAttachmentsMessage
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import { MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores';
	import { chatActivityStore } from '$lib/stores/chat/activity.svelte';

	const { Story } = defineMeta({
		args: {
			chatActions: noopChatActions,
			class: STORY_CLASS
		},
		component: ChatMessage,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessage'
	});

	const STREAM_ANSWER =
		"Based on my analysis, here's the solution:\n\n**Step 1:** Understand the requirements.\n\n**Step 2:** Implement the solution step by step.\n\n**Step 3:** Test and validate the results.";

	// Mutated by the streaming story's play function.
	let streamingMessage = $state<DatabaseMessage>({
		children: [],
		content: '',
		convId: CONV_ID,
		id: 'assistant-streaming',
		parent: userMessage.id,
		reasoningContent: '',
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: 'text'
	});

	async function streamInto(field: 'content' | 'reasoningContent', text: string) {
		for (let i = 0; i < text.length; i += 6) {
			streamingMessage[field] = text.slice(0, i + 6);
			await new Promise((resolve) => setTimeout(resolve, 30));
		}
	}
</script>

<Story args={{ message: assistantMessage }} beforeEach={() => seedChatStory()} name="Default" />

<Story args={{ message: userMessage }} beforeEach={() => seedChatStory()} name="User" />

<Story
	args={{ message: userMarkdownMessage }}
	beforeEach={() => seedChatStory()}
	name="UserMarkdown"
/>

<Story
	args={{ message: userMarkdownMessage }}
	beforeEach={() => seedChatStory({ settings: { renderUserContentAsMarkdown: false } })}
	name="UserPlainText"
/>

<Story
	args={{ message: userWithAttachmentsMessage }}
	beforeEach={() => seedChatStory()}
	name="UserWithAttachments"
/>

<Story
	args={{ message: userAttachmentsOnlyMessage }}
	beforeEach={() => seedChatStory()}
	name="UserAttachmentsOnly"
/>

<Story
	args={{ message: userMessage, nextAssistantMessage: assistantWithStatsMessage }}
	beforeEach={() => seedChatStory()}
	name="UserWithReadingStats"
/>

<Story
	args={{ isLastUserMessage: true, message: userMessage }}
	beforeEach={() => seedChatStory({ loading: true, processing: promptProcessingState })}
	name="UserReadingLive"
/>

<Story args={{ message: systemMessage }} beforeEach={() => seedChatStory()} name="System" />

<Story args={{ message: systemLongMessage }} beforeEach={() => seedChatStory()} name="SystemLong" />

<Story
	args={{ message: assistantMessage }}
	beforeEach={() => seedChatStory({ settings: { enableContinueGeneration: true } })}
	name="AssistantWithContinue"
/>

<Story
	args={{ message: assistantWithReasoningMessage }}
	beforeEach={() => seedChatStory()}
	name="AssistantWithReasoning"
/>

<Story
	args={{ message: assistantInterruptedReasoningMessage }}
	beforeEach={() => seedChatStory()}
	name="AssistantReasoningCancelled"
/>

<Story
	asChild
	beforeEach={() => seedChatStory({ loading: true, streamingMessageId: streamingMessage.id })}
	name="AssistantReasoningStreaming"
	play={async () => {
		streamingMessage.timestamp = 0;
		streamingMessage.content = '';
		streamingMessage.reasoningContent = '';

		await streamInto('reasoningContent', REASONING_TEXT);
		await streamInto('content', STREAM_ANSWER);

		chatStore.chatStreamingStates.delete(CONV_ID);
		chatActivityStore.localEnded(CONV_ID);
		streamingMessage.timestamp = Date.now();
	}}
>
	<ChatMessage
		chatActions={noopChatActions}
		class={STORY_CLASS}
		isLastAssistantMessage
		message={streamingMessage}
	/>
</Story>

<Story
	args={{
		isLastAssistantMessage: true,
		message: { ...assistantMessage, content: '', id: 'assistant-processing', timestamp: 0 }
	}}
	beforeEach={() => seedChatStory({ loading: true, processing: promptProcessingState })}
	name="AssistantProcessing"
/>

<Story
	args={{
		isLastAssistantMessage: true,
		message: {
			...assistantMessage,
			id: 'assistant-generating',
			model: assistantWithStatsMessage.model,
			timestamp: 0
		}
	}}
	beforeEach={() =>
		seedChatStory({
			loading: true,
			processing: generatingProcessingState,
			streamingMessageId: 'assistant-generating'
		})}
	name="AssistantGenerating"
/>

<Story
	args={{ message: assistantWithToolCallsMessage, toolMessages: toolResultMessages }}
	beforeEach={() => seedChatStory()}
	name="AssistantWithToolCalls"
/>

<Story
	args={{ message: assistantAgenticWithStatsMessage, toolMessages: agenticToolMessages }}
	beforeEach={() => seedChatStory({ settings: { showAgenticTurnStats: true } })}
	name="AssistantAgenticMultiTurn"
/>

<Story
	args={{ message: assistantWithStatsMessage }}
	beforeEach={() => seedChatStory()}
	name="AssistantWithStatistics"
/>

<Story
	args={{ message: rawOutputMessage }}
	beforeEach={() => seedChatStory({ settings: { showRawOutputSwitch: true } })}
	name="RawOutput"
	play={async ({ canvas, userEvent }) => {
		await userEvent.click(await canvas.findByRole('switch'));
	}}
/>

<Story
	args={{ message: assistantMessage, siblingInfo }}
	beforeEach={() => seedChatStory()}
	name="WithBranching"
/>

<Story
	args={{ message: compactionMessage }}
	beforeEach={() =>
		seedChatStory({ activeMessages: [preCompactionAssistantMessage, compactionMessage] })}
	name="Compaction"
/>

<Story
	args={{ message: compactionPendingMessage }}
	beforeEach={() =>
		seedChatStory({
			compactingNodeId: compactionPendingMessage.id,
			loading: true,
			processing: promptProcessingState
		})}
	name="CompactionPrefill"
/>

<Story
	args={{ message: compactionGeneratingMessage }}
	beforeEach={() =>
		seedChatStory({
			compactingNodeId: compactionGeneratingMessage.id,
			loading: true,
			processing: generatingProcessingState
		})}
	name="CompactionGenerating"
/>

<Story args={{ message: cwdChangeMessage }} beforeEach={() => seedChatStory()} name="CwdChange" />

<Story args={{ message: cwdClearedMessage }} beforeEach={() => seedChatStory()} name="CwdCleared" />

<Story args={{ message: mcpPromptMessage }} beforeEach={() => seedChatStory()} name="McpPrompt" />

<Story
	args={{ message: userWithAttachmentsMessage }}
	beforeEach={() => seedChatStory({ pendingEditMessageId: userWithAttachmentsMessage.id })}
	name="EditUser"
/>

<Story
	args={{ message: assistantWithReasoningMessage }}
	beforeEach={() => seedChatStory({ pendingEditMessageId: assistantWithReasoningMessage.id })}
	name="EditAssistant"
/>

<Story
	args={{ message: systemMessage }}
	beforeEach={() => seedChatStory({ pendingEditMessageId: systemMessage.id })}
	name="EditSystem"
/>
