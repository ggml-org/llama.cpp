<script lang="ts">
	// A basic chat reply: the model's reasoning, if any, and its answer. A reply
	// that ran the agentic loop renders as an Agent Trace (ChatMessageAgentTrace)
	// instead, with its tool calls grouped into rows.

	import ChatMessageReasoningBlock from '../../ChatMessageReasoningBlock.svelte';
	import ChatMessageAssistantProcessingInfo from './ChatMessageAssistantProcessingInfo.svelte';
	import { MarkdownContent } from '#lib/components/app/index.ts';
	import { REPLY_STATUS_FALLBACK } from '#lib/constants/index.ts';
	import { AgenticSectionType } from '#lib/enums/index.ts';
	import { agenticStore, settingsStore } from '#lib/stores/index.ts';
	import type { AgenticSection, DatabaseMessage } from '#lib/types/index.ts';
	import {
		deriveAgenticSections,
		isSectionExpandedByDefault,
		showsReplyStatus
	} from '#lib/utils/index.ts';

	interface Props {
		message: DatabaseMessage;
		isStreaming?: boolean;
		isLastAssistantMessage?: boolean;
		/**
		 * What the server is doing while this reply waits on it ( "Processing
		 * 40%" ), empty while tokens stream, null when the reply is not live.
		 */
		statusText?: string | null;
	}

	let {
		isLastAssistantMessage = false,
		isStreaming: isChatStreaming = false,
		message,
		statusText = null
	}: Props = $props();

	// only the last assistant message is live; the chat-level flag is shared
	const isStreaming = $derived(isChatStreaming && isLastAssistantMessage);

	let expandedStates: Record<number, boolean> = $state({});

	const showThoughtInProgress = $derived(Boolean(settingsStore.config.showThoughtInProgress));

	const hasReasoningError = $derived(
		isLastAssistantMessage ? !!agenticStore.getLastError(message.convId) : false
	);

	const sections = $derived(deriveAgenticSections(message, [], [], isStreaming));

	function isExpanded(index: number, section: AgenticSection): boolean {
		if (expandedStates[index] !== undefined) return expandedStates[index];

		// a basic reply has no tool sections, so the two tool settings never apply
		return isSectionExpandedByDefault(section, {
			alwaysShowToolCallContent: false,
			showThoughtInProgress,
			showToolCallInProgress: false
		});
	}

	function toggleExpanded(index: number, section: AgenticSection) {
		expandedStates[index] = !isExpanded(index, section);
	}

	const showStatus = $derived(showsReplyStatus(statusText, sections.at(-1), isStreaming));
</script>

<div class="assistant-response">
	{#each sections as section, index (index)}
		{#if section.type === AgenticSectionType.TEXT}
			<div class="assistant-response-text">
				<MarkdownContent attachments={message?.extra} content={section.content} />
			</div>
		{:else if section.type === AgenticSectionType.REASONING || section.type === AgenticSectionType.REASONING_PENDING}
			<ChatMessageReasoningBlock
				attachments={message?.extra}
				{hasReasoningError}
				{isStreaming}
				onToggle={() => toggleExpanded(index, section)}
				open={isExpanded(index, section)}
				{section}
			/>
		{/if}
	{/each}

	{#if showStatus}
		<ChatMessageAssistantProcessingInfo text={statusText || REPLY_STATUS_FALLBACK} />
	{/if}
</div>

<style>
	.assistant-response {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		max-width: 48rem;
	}

	.assistant-response > :global(*) {
		min-width: 0;
	}

	/* the answer stands a little apart from the reasoning above it, as in a trace */
	.assistant-response > :global(:not(.assistant-response-text) + .assistant-response-text) {
		margin-top: 0.25rem;
	}
</style>
