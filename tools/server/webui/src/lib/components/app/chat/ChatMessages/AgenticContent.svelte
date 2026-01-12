<script lang="ts">
	/**
	 * AgenticContent - Chronological display of agentic flow output
	 *
	 * Parses content with tool call markers and displays them inline
	 * with text content. Each tool call is shown as a collapsible box
	 * similar to the reasoning/thinking block UI.
	 */

	import {
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { agenticStreamingToolCall } from '$lib/stores/agentic.svelte';
	import { Wrench, Loader2 } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import { AGENTIC_TAGS, AGENTIC_REGEX } from '$lib/constants/agentic';
	import { formatJsonPretty } from '$lib/utils/formatters';
	import { decodeBase64 } from '$lib/utils';
	import type { ChatMessageToolCallTiming } from '$lib/types/chat';

	interface Props {
		content: string;
		isStreaming?: boolean;
		toolCallTimings?: ChatMessageToolCallTiming[];
	}

	interface AgenticSection {
		type: AgenticSectionType;
		content: string;
		toolName?: string;
		toolArgs?: string;
		toolResult?: string;
	}

	let { content, isStreaming = false, toolCallTimings = [] }: Props = $props();

	const sections = $derived(parseAgenticContent(content));

	// Get timing for a specific tool call by index (completed tool calls only)
	function getToolCallTiming(toolCallIndex: number): ChatMessageToolCallTiming | undefined {
		return toolCallTimings[toolCallIndex];
	}

	// Get streaming tool call from reactive store (not from content markers)
	const streamingToolCall = $derived(isStreaming ? agenticStreamingToolCall() : null);

	let expandedStates: Record<number, boolean> = $state({});

	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);

	function getDefaultExpanded(isPending: boolean): boolean {
		return showToolCallInProgress && isPending;
	}

	function isExpanded(index: number, isPending: boolean): boolean {
		if (expandedStates[index] !== undefined) {
			return expandedStates[index];
		}
		return getDefaultExpanded(isPending);
	}

	function toggleExpanded(index: number, isPending: boolean) {
		const currentState = isExpanded(index, isPending);
		expandedStates[index] = !currentState;
	}

	function parseAgenticContent(rawContent: string): AgenticSection[] {
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
			const toolArgsBase64 = match[2];
			const toolArgs = decodeBase64(toolArgsBase64);
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
			const toolArgsBase64 = pendingMatch[2];
			const toolArgs = decodeBase64(toolArgsBase64);
			// Capture streaming result content (everything after args marker)
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

			const partialArgsBase64 = partialWithNameMatch[2] || '';
			const partialArgs = decodeBase64(partialArgsBase64);

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
</script>

<div class="agentic-content">
	{#each sections as section, index (index)}
		{#if section.type === AgenticSectionType.TEXT}
			<div class="agentic-text">
				<MarkdownContent content={section.content} />
			</div>
		{:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING}
			{@const isPending = section.type === AgenticSectionType.TOOL_CALL_PENDING}
			{@const toolIcon = isPending ? Loader2 : Wrench}
			{@const toolIconClass = isPending ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}
			{@const toolCallIndex =
				sections.slice(0, index + 1).filter((s) => s.type === AgenticSectionType.TOOL_CALL).length -
				1}
			{@const timing = !isPending ? getToolCallTiming(toolCallIndex) : undefined}
			<CollapsibleContentBlock
				open={isExpanded(index, isPending)}
				class="my-2"
				icon={toolIcon}
				iconClass={toolIconClass}
				title={section.toolName || ''}
				subtitle={isPending
					? 'executing...'
					: timing
						? `${(timing.duration_ms / 1000).toFixed(2)}s`
						: undefined}
				onToggle={() => toggleExpanded(index, isPending)}
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
