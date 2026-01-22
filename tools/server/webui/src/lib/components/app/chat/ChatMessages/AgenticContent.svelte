<script lang="ts">
	/**
	 * AgenticContent - Chronological display of agentic flow output
	 *
	 * Parses content with tool call and reasoning markers and displays them inline
	 * with text content. Tool calls and reasoning are shown as collapsible blocks.
	 */

	import {
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { Wrench, Loader2, AlertTriangle, Brain } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import { AGENTIC_TAGS, AGENTIC_REGEX, REASONING_TAGS } from '$lib/constants/agentic';
	import { formatJsonPretty } from '$lib/utils/formatters';
	import type { DatabaseMessage } from '$lib/types/database';

	interface Props {
		message?: DatabaseMessage;
		content: string;
		isStreaming?: boolean;
	}

	interface AgenticSection {
		type: AgenticSectionType;
		content: string;
		toolName?: string;
		toolArgs?: string;
		toolResult?: string;
	}

	let { content, message, isStreaming = false }: Props = $props();

	const sections = $derived(parseAgenticContent(content));

	// Get timing for a specific tool call by index (completed tool calls only)
	function getToolCallTiming(toolCallIndex: number): ChatMessageToolCallTiming | undefined {
		return toolCallTimings[toolCallIndex];
	}

	// Get streaming tool call from reactive store (not from content markers)
	const streamingToolCall = $derived(isStreaming ? agenticStreamingToolCall() : null);

	let expandedStates: Record<number, boolean> = $state({});

	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);
	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	function getDefaultExpanded(section: AgenticSection): boolean {
		if (
			section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING
		) {
			return showToolCallInProgress;
		}
		if (section.type === AgenticSectionType.REASONING_PENDING) {
			return showThoughtInProgress;
		}
		return false;
	}

	function isExpanded(index: number, section: AgenticSection): boolean {
		if (expandedStates[index] !== undefined) {
			return expandedStates[index];
		}
		return getDefaultExpanded(section);
	}

	function toggleExpanded(index: number, section: AgenticSection) {
		const currentState = isExpanded(index, section);
		expandedStates[index] = !currentState;
	}

	type ReasoningSegment = {
		type: 'text' | 'reasoning' | 'reasoning_pending';
		content: string;
	};

	function stripPartialMarker(text: string): string {
		const partialMarkerMatch = text.match(AGENTIC_REGEX.PARTIAL_MARKER);
		if (partialMarkerMatch) {
			return text.slice(0, partialMarkerMatch.index).trim();
		}
		return text;
	}

	function splitReasoningSegments(rawContent: string): ReasoningSegment[] {
		if (!rawContent) return [];

		const segments: ReasoningSegment[] = [];
		let cursor = 0;

		while (cursor < rawContent.length) {
			const startIndex = rawContent.indexOf(REASONING_TAGS.START, cursor);
			if (startIndex === -1) {
				const remainingText = rawContent.slice(cursor);
				if (remainingText) {
					segments.push({ type: 'text', content: remainingText });
				}
				break;
			}

			if (startIndex > cursor) {
				const textBefore = rawContent.slice(cursor, startIndex);
				if (textBefore) {
					segments.push({ type: 'text', content: textBefore });
				}
			}

			const contentStart = startIndex + REASONING_TAGS.START.length;
			const endIndex = rawContent.indexOf(REASONING_TAGS.END, contentStart);

			if (endIndex === -1) {
				const pendingContent = rawContent.slice(contentStart);
				segments.push({
					type: 'reasoning_pending',
					content: stripPartialMarker(pendingContent)
				});
				break;
			}

			const reasoningContent = rawContent.slice(contentStart, endIndex);
			segments.push({ type: 'reasoning', content: reasoningContent });
			cursor = endIndex + REASONING_TAGS.END.length;
		}

		return segments;
	}

	function parseToolCallContent(rawContent: string): AgenticSection[] {
		if (!rawContent) return [];

		const sections: AgenticSection[] = [];

		const completedToolCallRegex = new RegExp(AGENTIC_REGEX.COMPLETED_TOOL_CALL.source, 'g');

		let lastIndex = 0;
		let match;

		while ((match = completedToolCallRegex.exec(rawContent)) !== null) {
			if (match.index > lastIndex) {
				const textBefore = rawContent.slice(lastIndex, match.index).trim();
				if (textBefore) {
					sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
				}
			}

			const toolName = match[1];
			const toolArgs = match[2]; // Direct JSON
			const toolResult = match[3].replace(/^\n+|\n+$/g, '');

			sections.push({
				type: AgenticSectionType.TOOL_CALL,
				content: toolResult,
				toolName,
				toolArgs,
				toolResult
			});

			lastIndex = match.index + match[0].length;
		}

		const remainingContent = rawContent.slice(lastIndex);

		const pendingMatch = remainingContent.match(AGENTIC_REGEX.PENDING_TOOL_CALL);

		const partialWithNameMatch = remainingContent.match(AGENTIC_REGEX.PARTIAL_WITH_NAME);

		const earlyMatch = remainingContent.match(AGENTIC_REGEX.EARLY_MATCH);

		if (pendingMatch) {
			const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
				}
			}

			const toolName = pendingMatch[1];
			const toolArgs = pendingMatch[2]; // Direct JSON
			// Capture streaming result content (everything after TOOL_ARGS_END marker)
			const streamingResult = (pendingMatch[3] || '').replace(/^\n+|\n+$/g, '');

			sections.push({
				type: AgenticSectionType.TOOL_CALL_PENDING,
				content: streamingResult,
				toolName,
				toolArgs,
				toolResult: streamingResult || undefined
			});
		} else if (partialWithNameMatch) {
			const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
				}
			}

			const partialArgs = partialWithNameMatch[2] || ''; // Direct JSON streaming

			sections.push({
				type: AgenticSectionType.TOOL_CALL_STREAMING,
				content: '',
				toolName: partialWithNameMatch[1],
				toolArgs: partialArgs || undefined,
				toolResult: undefined
			});
		} else if (earlyMatch) {
			// Just START marker, show streaming state
			const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
				}
			}

			// Try to extract tool name if present
			const nameMatch = earlyMatch[1]?.match(AGENTIC_REGEX.TOOL_NAME_EXTRACT);

			sections.push({
				type: AgenticSectionType.TOOL_CALL_STREAMING,
				content: '',
				toolName: nameMatch?.[1],
				toolArgs: undefined,
				toolResult: undefined
			});
		} else if (lastIndex < rawContent.length) {
			// Add remaining text after last completed tool call
			// But strip any partial markers that might be starting
			let remainingText = rawContent.slice(lastIndex).trim();

			// Check for partial marker at the end (e.g., "<<<" or "<<<AGENTIC" etc.)
			const partialMarkerMatch = remainingText.match(AGENTIC_REGEX.PARTIAL_MARKER);
			if (partialMarkerMatch) {
				remainingText = remainingText.slice(0, partialMarkerMatch.index).trim();
			}

			if (remainingText) {
				sections.push({ type: AgenticSectionType.TEXT, content: remainingText });
			}
		}

		// If no tool calls found, return content as single text section
		if (sections.length === 0 && rawContent.trim()) {
			sections.push({ type: AgenticSectionType.TEXT, content: rawContent });
		}

		return sections;
	}

	function parseAgenticContent(rawContent: string): AgenticSection[] {
		if (!rawContent) return [];

		const segments = splitReasoningSegments(rawContent);
		const sections: AgenticSection[] = [];

		for (const segment of segments) {
			if (segment.type === 'text') {
				sections.push(...parseToolCallContent(segment.content));
				continue;
			}

			if (segment.type === 'reasoning') {
				if (segment.content.trim()) {
					sections.push({ type: AgenticSectionType.REASONING, content: segment.content });
				}
				continue;
			}

			sections.push({
				type: AgenticSectionType.REASONING_PENDING,
				content: segment.content
			});
		}

		return sections;
	}
</script>

<div class="agentic-content">
	{#each sections as section, index (index)}
		{#if section.type === AgenticSectionType.TEXT}
			<div class="agentic-text">
				<MarkdownContent content={section.content} {message} />
			</div>
		{:else if section.type === AgenticSectionType.TOOL_CALL_STREAMING}
			{@const streamingIcon = isStreaming ? Loader2 : AlertTriangle}
			{@const streamingIconClass = isStreaming ? 'h-4 w-4 animate-spin' : 'h-4 w-4 text-yellow-500'}
			{@const streamingSubtitle = isStreaming ? 'streaming...' : 'incomplete'}
			<CollapsibleContentBlock
				open={isExpanded(index, section)}
				class="my-2"
				icon={streamingIcon}
				iconClass={streamingIconClass}
				title={section.toolName || 'Tool call'}
				subtitle={streamingSubtitle}
				{isStreaming}
				onToggle={() => toggleExpanded(index, section)}
			>
				<div class="pt-3">
					<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
						<span>Arguments:</span>
						{#if isStreaming}
							<Loader2 class="h-3 w-3 animate-spin" />
						{/if}
					</div>
					{#if section.toolArgs}
						<SyntaxHighlightedCode
							code={formatJsonPretty(section.toolArgs)}
							language="json"
							maxHeight="20rem"
							class="text-xs"
						/>
					{:else if isStreaming}
						<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
							Receiving arguments...
						</div>
					{:else}
						<div
							class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
						>
							Response was truncated
						</div>
					{/if}
				</div>
			</CollapsibleContentBlock>
		{:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING}
			{@const isPending = section.type === AgenticSectionType.TOOL_CALL_PENDING}
			{@const toolIcon = isPending ? Loader2 : Wrench}
			{@const toolIconClass = isPending ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}
			{@const toolCallIndex =
				sections.slice(0, index + 1).filter((s) => s.type === AgenticSectionType.TOOL_CALL).length -
				1}
			{@const timing = !isPending ? getToolCallTiming(toolCallIndex) : undefined}
			<CollapsibleContentBlock
				open={isExpanded(index, section)}
				class="my-2"
				icon={toolIcon}
				iconClass={toolIconClass}
				title={section.toolName || ''}
				subtitle={isPending ? 'executing...' : undefined}
				isStreaming={isPending}
				onToggle={() => toggleExpanded(index, section)}
			>
				{#if section.toolArgs && section.toolArgs !== '{}'}
					<div class="pt-3">
						<div class="my-3 text-xs text-muted-foreground">Arguments:</div>
						<SyntaxHighlightedCode
							code={formatJsonPretty(section.toolArgs)}
							language="json"
							maxHeight="20rem"
							class="text-xs"
						/>
					</div>
				{/if}

				<div class="pt-3">
					<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
						<span>Result:</span>
						{#if isPending}
							<Loader2 class="h-3 w-3 animate-spin" />
						{/if}
					</div>
					{#if section.toolResult}
						<div class="overflow-auto rounded-lg border border-border bg-muted">
							<!-- prettier-ignore -->
							<pre class="m-0 overflow-x-auto whitespace-pre-wrap p-4 font-mono text-xs leading-relaxed"><code>{section.toolResult}</code></pre>
						</div>
					{:else if isPending}
						<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
							Waiting for result...
						</div>
					{/if}
				</div>
			</CollapsibleContentBlock>
		{:else if section.type === AgenticSectionType.REASONING}
			<CollapsibleContentBlock
				open={isExpanded(index, section)}
				class="my-2"
				icon={Brain}
				title="Reasoning"
				onToggle={() => toggleExpanded(index, section)}
			>
				<div class="pt-3">
					<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
						{section.content}
					</div>
				</div>
			</CollapsibleContentBlock>
		{:else if section.type === AgenticSectionType.REASONING_PENDING}
			{@const reasoningTitle = isStreaming ? 'Reasoning...' : 'Reasoning'}
			{@const reasoningSubtitle = isStreaming ? 'streaming...' : 'incomplete'}
			<CollapsibleContentBlock
				open={isExpanded(index, section)}
				class="my-2"
				icon={Brain}
				title={reasoningTitle}
				subtitle={reasoningSubtitle}
				{isStreaming}
				onToggle={() => toggleExpanded(index, section)}
			>
				<div class="pt-3">
					<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
						{section.content}
					</div>
				</div>
			</CollapsibleContentBlock>
		{/if}
	{/each}

	{#if streamingToolCall}
		<CollapsibleContentBlock
			open={true}
			class="my-2"
			icon={Loader2}
			iconClass="h-4 w-4 animate-spin"
			title={streamingToolCall.name || 'Tool call'}
			subtitle="streaming..."
			onToggle={() => {}}
		>
			<div class="pt-3">
				<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
					<span>Arguments:</span>
					<Loader2 class="h-3 w-3 animate-spin" />
				</div>
				{#if streamingToolCall.arguments}
					<SyntaxHighlightedCode
						code={formatJsonPretty(streamingToolCall.arguments)}
						language="json"
						maxHeight="20rem"
						class="text-xs"
					/>
				{:else}
					<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
						Receiving arguments...
					</div>
				{/if}
			</div>
		</CollapsibleContentBlock>
	{/if}
</div>

<style>
	.agentic-content {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		max-width: 48rem;
	}

	.agentic-text {
		width: 100%;
	}
</style>
