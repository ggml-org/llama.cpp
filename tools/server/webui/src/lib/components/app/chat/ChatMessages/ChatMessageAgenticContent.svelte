<script lang="ts">
	import {
		ChatMessageStatistics,
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { Wrench, Loader2, AlertTriangle, Brain } from '@lucide/svelte';
	import { AgenticSectionType, AttachmentType, FileTypeText } from '$lib/enums';
	import { formatJsonPretty } from '$lib/utils';
	import { ATTACHMENT_SAVED_REGEX, NEWLINE_SEPARATOR } from '$lib/constants';
	import { getJavaScriptSourceArgument } from '$lib/services/tools/codeInterpreterSharedState';
	import { parseAgenticContent, type AgenticSection } from '$lib/utils';
	import type { DatabaseMessage, DatabaseMessageExtraImageFile } from '$lib/types/database';
	import type { ChatMessageAgenticTimings, ChatMessageAgenticTurnStats } from '$lib/types/chat';
	import { ChatMessageStatsView } from '$lib/enums';

	interface Props {
		message?: DatabaseMessage;
		content: string;
		isStreaming?: boolean;
		highlightTurns?: boolean;
	}

	type ToolResultLine = {
		text: string;
		image?: DatabaseMessageExtraImageFile;
	};

	type ParsedAgenticSection = AgenticSection & {
		parsedLines: ToolResultLine[];
	};

	type InlineFlowGroup = {
		kind: 'flow';
		flatIndices: number[];
		sections: ParsedAgenticSection[];
	};

	type InlineTextGroup = {
		kind: 'text';
		flatIndex: number;
		section: ParsedAgenticSection;
	};

	type InlineRenderGroup = InlineFlowGroup | InlineTextGroup;

	let { content, message, isStreaming = false, highlightTurns = false }: Props = $props();

	let expandedStates: Record<number, boolean> = $state({});
	let autoExpandedStates: Record<number, boolean> = $state({});

	const sections = $derived(parseAgenticContent(content));
	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);
	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	// Parse toolResults with images only when sections or message.extra change
	const sectionsParsed = $derived<ParsedAgenticSection[]>(
		sections.map((section) => ({
			...section,
			parsedLines: section.toolResult
				? parseToolResultWithImages(section.toolResult, message?.extra)
				: []
		}))
	);

	function isToolSection(section: ParsedAgenticSection): boolean {
		return (
			section.type === AgenticSectionType.TOOL_CALL ||
			section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING
		);
	}

	function isReasoningSection(section: ParsedAgenticSection): boolean {
		return (
			section.type === AgenticSectionType.REASONING ||
			section.type === AgenticSectionType.REASONING_PENDING
		);
	}

	function isPendingFlowSection(section: ParsedAgenticSection): boolean {
		return (
			section.type === AgenticSectionType.REASONING_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING
		);
	}

	function buildInlineRenderGroups(
		parsedSections: ParsedAgenticSection[],
		flatIndices: number[]
	): InlineRenderGroup[] {
		const groups: InlineRenderGroup[] = [];
		let currentFlowSections: ParsedAgenticSection[] = [];
		let currentFlowIndices: number[] = [];

		const flushFlow = () => {
			if (currentFlowSections.length === 0) {
				return;
			}

			groups.push({
				kind: 'flow',
				sections: currentFlowSections,
				flatIndices: currentFlowIndices
			});
			currentFlowSections = [];
			currentFlowIndices = [];
		};

		for (let i = 0; i < parsedSections.length; i++) {
			const section = parsedSections[i];
			const flatIndex = flatIndices[i];

			if (section.type === AgenticSectionType.TEXT) {
				flushFlow();
				groups.push({
					kind: 'text',
					section,
					flatIndex
				});
				continue;
			}

			currentFlowSections.push(section);
			currentFlowIndices.push(flatIndex);
		}

		flushFlow();

		return groups;
	}

	function isFlowGroup(group: InlineRenderGroup): group is InlineFlowGroup {
		return group.kind === 'flow';
	}

	const inlineGroups = $derived.by(() =>
		buildInlineRenderGroups(
			sectionsParsed,
			sectionsParsed.map((_, index) => index)
		)
	);

	// Group flat sections into agentic turns
	// A new turn starts when a non-tool section follows a tool section
	const turnGroups = $derived.by(() => {
		const turns: { sections: (typeof sectionsParsed)[number][]; flatIndices: number[] }[] = [];
		let currentTurn: (typeof sectionsParsed)[number][] = [];
		let currentIndices: number[] = [];
		let prevWasTool = false;

		for (let i = 0; i < sectionsParsed.length; i++) {
			const section = sectionsParsed[i];
			const isTool =
				section.type === AgenticSectionType.TOOL_CALL ||
				section.type === AgenticSectionType.TOOL_CALL_PENDING ||
				section.type === AgenticSectionType.TOOL_CALL_STREAMING;

			if (!isTool && prevWasTool && currentTurn.length > 0) {
				turns.push({ sections: currentTurn, flatIndices: currentIndices });
				currentTurn = [];
				currentIndices = [];
			}

			currentTurn.push(section);
			currentIndices.push(i);
			prevWasTool = isTool;
		}

		if (currentTurn.length > 0) {
			turns.push({ sections: currentTurn, flatIndices: currentIndices });
		}

		return turns;
	});

	const renderedGroups = $derived.by(() => {
		if (highlightTurns && turnGroups.length > 1) {
			return turnGroups.flatMap((turn) => buildInlineRenderGroups(turn.sections, turn.flatIndices));
		}

		return inlineGroups;
	});

	const renderedFlowGroups = $derived.by(() => renderedGroups.filter(isFlowGroup));

	function getDefaultExpanded(group: InlineFlowGroup): boolean {
		return group.sections.some((section) => {
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
		});
	}

	function shouldPreserveExpandedState(group: InlineFlowGroup): boolean {
		const lastFlatIndex = Math.max(...group.flatIndices);
		const hasInlineTextAfterGroup = sectionsParsed
			.slice(lastFlatIndex + 1)
			.some((section) => section.type === AgenticSectionType.TEXT);

		return (
			group.sections.some((section) => isReasoningSection(section)) &&
			group.sections.some((section) => isToolSection(section)) &&
			!hasInlineTextAfterGroup
		);
	}

	function isExpanded(group: InlineFlowGroup): boolean {
		const groupKey = group.flatIndices[0];

		if (expandedStates[groupKey] !== undefined) {
			return expandedStates[groupKey];
		}

		if (autoExpandedStates[groupKey] !== undefined) {
			return autoExpandedStates[groupKey];
		}

		return getDefaultExpanded(group);
	}

	function toggleExpanded(group: InlineFlowGroup) {
		const groupKey = group.flatIndices[0];
		const currentState = isExpanded(group);

		expandedStates[groupKey] = !currentState;
	}

	$effect(() => {
		const activeGroupKeys = new Set<number>();

		for (const group of renderedFlowGroups) {
			const groupKey = group.flatIndices[0];
			activeGroupKeys.add(groupKey);

			if (expandedStates[groupKey] !== undefined) {
				continue;
			}

			if (autoExpandedStates[groupKey] === undefined) {
				autoExpandedStates[groupKey] = getDefaultExpanded(group);
				continue;
			}

			if (!shouldPreserveExpandedState(group)) {
				autoExpandedStates[groupKey] = getDefaultExpanded(group);
			}
		}

		for (const key of Object.keys(autoExpandedStates)) {
			const numericKey = Number(key);
			if (!activeGroupKeys.has(numericKey)) {
				delete autoExpandedStates[numericKey];
			}
		}
	});

	function parseToolResultWithImages(
		toolResult: string,
		extras?: DatabaseMessage['extra']
	): ToolResultLine[] {
		const lines = toolResult.split(NEWLINE_SEPARATOR);

		return lines.map((line) => {
			const match = line.match(ATTACHMENT_SAVED_REGEX);
			if (!match || !extras) return { text: line };

			const attachmentName = match[1];
			const image = extras.find(
				(e): e is DatabaseMessageExtraImageFile =>
					e.type === AttachmentType.IMAGE && e.name === attachmentName
			);

			return { text: line, image };
		});
	}

	function buildTurnAgenticTimings(stats: ChatMessageAgenticTurnStats): ChatMessageAgenticTimings {
		return {
			turns: 1,
			toolCallsCount: stats.toolCalls.length,
			toolsMs: stats.toolsMs,
			toolCalls: stats.toolCalls,
			llm: stats.llm
		};
	}

	function getToolArgumentsCode(section: ParsedAgenticSection): string | undefined {
		if (!section.toolArgs) {
			return undefined;
		}

		return getJavaScriptSourceArgument(section.toolName, section.toolArgs);
	}
	</script>

{#snippet renderInlineToolSection(section: ParsedAgenticSection)}
	{@const isPending = section.type === AgenticSectionType.TOOL_CALL_PENDING}
	{@const isStreamingTool = section.type === AgenticSectionType.TOOL_CALL_STREAMING}
	{@const ToolIcon = isPending
		? Loader2
		: isStreamingTool
			? isStreaming
				? Loader2
				: AlertTriangle
			: Wrench}
	{@const toolIconClass = isPending || (isStreamingTool && isStreaming)
		? 'h-4 w-4 animate-spin'
		: isStreamingTool
			? 'h-4 w-4 text-yellow-500'
			: 'h-4 w-4'}

	<div class="agentic-inline-tool" data-testid="inline-tool-call">
		<div class="agentic-inline-tool-header">
			<div class="flex items-center gap-2">
				<ToolIcon class={toolIconClass} />
				<span class="font-mono text-xs font-medium">
					{section.toolName || 'Tool call'}
				</span>
			</div>

			{#if isPending}
				<span class="text-[11px] text-muted-foreground italic">executing...</span>
			{:else if isStreamingTool && !isStreaming}
				<span class="text-[11px] text-muted-foreground italic">incomplete</span>
			{/if}
		</div>

			{#if section.toolArgs && section.toolArgs !== '{}'}
				{@const toolArgumentsCode = getToolArgumentsCode(section)}
				<div class="agentic-inline-label">Arguments</div>
				{#if toolArgumentsCode}
					<div data-testid="inline-tool-args-code">
						<SyntaxHighlightedCode
							code={toolArgumentsCode}
							language={FileTypeText.JAVASCRIPT}
							maxHeight="20rem"
							class="text-xs"
						/>
					</div>
				{:else}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolArgs)}
						language={FileTypeText.JSON}
						maxHeight="20rem"
						class="text-xs"
					/>
				{/if}
			{:else if isStreamingTool}
				<div class="agentic-inline-label">Arguments</div>
			{#if isStreaming}
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
		{/if}

		{#if section.toolResult || isPending}
			<div class="agentic-inline-label">Result</div>
			{#if section.toolResult}
				<div class="overflow-auto rounded-lg border border-border bg-muted p-4">
					{#each section.parsedLines as line, i (i)}
						<div class="font-mono text-xs leading-relaxed whitespace-pre-wrap">{line.text}</div>
						{#if line.image}
							<img
								src={line.image.base64Url}
								alt={line.image.name}
								class="mt-2 mb-2 h-auto max-w-full rounded-lg"
								loading="lazy"
							/>
						{/if}
					{/each}
				</div>
			{:else}
				<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
					Waiting for result...
				</div>
			{/if}
		{/if}
	</div>
{/snippet}

{#snippet renderFlowGroup(group: InlineFlowGroup)}
	{@const hasReasoning = group.sections.some((section) => isReasoningSection(section))}
	{@const isPendingFlow = group.sections.some((section) => isPendingFlowSection(section))}
	{@const groupTitle = hasReasoning ? (isPendingFlow && isStreaming ? 'Reasoning...' : 'Reasoning') : 'Tool call'}
	{@const groupSubtitle = hasReasoning && isPendingFlow && !isStreaming ? 'incomplete' : undefined}
	{@const groupIcon = hasReasoning ? Brain : Wrench}

	<CollapsibleContentBlock
		open={isExpanded(group)}
		class="my-2"
		icon={groupIcon}
		title={groupTitle}
		subtitle={groupSubtitle}
		isStreaming={isPendingFlow}
		onToggle={() => toggleExpanded(group)}
	>
		<div class="agentic-inline-flow">
			{#each group.sections as section, groupIndex (group.flatIndices[groupIndex])}
				{#if groupIndex > 0}
					<div class="agentic-flow-divider"></div>
				{/if}

				{#if isReasoningSection(section)}
					<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
						{section.content}
					</div>
				{:else if isToolSection(section)}
					{@render renderInlineToolSection(section)}
				{/if}
			{/each}
		</div>
	</CollapsibleContentBlock>
{/snippet}

{#snippet renderGroup(group: InlineRenderGroup)}
	{#if group.kind === 'text'}
		<div class="agentic-text">
			<MarkdownContent content={group.section.content} attachments={message?.extra} />
		</div>
	{:else}
		{@render renderFlowGroup(group)}
	{/if}
{/snippet}

<div class="agentic-content">
	{#if highlightTurns && turnGroups.length > 1}
		{#each turnGroups as turn, turnIndex (turnIndex)}
			{@const turnStats = message?.timings?.agentic?.perTurn?.[turnIndex]}
			{@const turnGroupsInline = buildInlineRenderGroups(turn.sections, turn.flatIndices)}
			<div class="agentic-turn my-2 hover:bg-muted/80 dark:hover:bg-muted/30">
				<span class="agentic-turn-label">Turn {turnIndex + 1}</span>
				{#each turnGroupsInline as group, groupIndex (`${turnIndex}-${groupIndex}`)}
					{@render renderGroup(group)}
				{/each}
				{#if turnStats}
					<div class="turn-stats">
						<ChatMessageStatistics
							promptTokens={turnStats.llm.prompt_n}
							promptMs={turnStats.llm.prompt_ms}
							predictedTokens={turnStats.llm.predicted_n}
							predictedMs={turnStats.llm.predicted_ms}
							agenticTimings={turnStats.toolCalls.length > 0
								? buildTurnAgenticTimings(turnStats)
								: undefined}
							initialView={ChatMessageStatsView.GENERATION}
							hideSummary
						/>
					</div>
				{/if}
			</div>
		{/each}
	{:else}
		{#each inlineGroups as group, groupIndex (groupIndex)}
			{@render renderGroup(group)}
		{/each}
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

	.agentic-inline-flow {
		display: flex;
		flex-direction: column;
		gap: 0.875rem;
		padding-top: 0.75rem;
	}

	.agentic-flow-divider {
		border-top: 1px solid hsl(var(--muted) / 0.75);
	}

	.agentic-inline-tool {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.875rem;
		border: 1px solid hsl(var(--border) / 0.75);
		border-radius: 0.75rem;
		background: hsl(var(--muted) / 0.35);
	}

	.agentic-inline-tool-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
	}

	.agentic-inline-label {
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--muted-foreground);
	}

	.agentic-turn {
		position: relative;
		border: 1.5px dashed var(--muted-foreground);
		border-radius: 0.75rem;
		padding: 1rem;
		transition: background 0.1s;
	}

	.agentic-turn-label {
		position: absolute;
		top: -1rem;
		left: 0.75rem;
		padding: 0 0.375rem;
		background: var(--background);
		font-size: 0.7rem;
		font-weight: 500;
		color: var(--muted-foreground);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.turn-stats {
		margin-top: 0.75rem;
		padding-top: 0.5rem;
		border-top: 1px solid hsl(var(--muted) / 0.5);
	}
</style>
