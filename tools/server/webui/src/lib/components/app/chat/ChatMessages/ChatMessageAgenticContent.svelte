<script lang="ts">
	import {
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { Wrench, Loader2, AlertTriangle, Brain } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import { formatJsonPretty } from '$lib/utils';
	import { parseAgenticContent, type AgenticSection } from '$lib/utils/agentic';
	import type { DatabaseMessage } from '$lib/types/database';

	interface Props {
		/** Optional database message for context */
		message?: DatabaseMessage;
		/** Raw content string to parse and display */
		content: string;
		/** Whether content is currently streaming */
		isStreaming?: boolean;
	}

	let { content, message, isStreaming = false }: Props = $props();

	let expandedStates: Record<number, boolean> = $state({});

	const sections = $derived(parseAgenticContent(content));
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
