<script lang="ts">
	import { Wrench, Loader2, Lightbulb } from '@lucide/svelte';
	import {
		ChatMessageStatistics,
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode,
		ChatMessageActionCardPermissionRequest,
		ChatMessageActionCardContinueRequest
	} from '$lib/components/app';

	import {
		AgenticSectionType,
		ChatMessageStatsView,
		FileTypeText,
		ToolPermissionDecision
	} from '$lib/enums';
	import type {
		ChatMessageAgenticTimings,
		ChatMessageAgenticTurnStats,
		DatabaseMessage
	} from '$lib/types';
	import {
		deriveAgenticSections,
		formatJsonPretty,
		getFileTypeByExtension,
		parseToolResultWithImages,
		type AgenticSection,
		type ToolResultLine
	} from '$lib/utils';
	import {
		agenticPendingPermissionRequest,
		agenticResolvePermission,
		agenticPendingContinueRequest,
		agenticResolveContinue,
		agenticLastError
	} from '$lib/stores/agentic.svelte';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		isStreaming?: boolean;
		isLastAssistantMessage?: boolean;
	}

	let {
		message,
		toolMessages = [],
		isStreaming = false,
		isLastAssistantMessage = false
	}: Props = $props();

	let expandedStates: Record<number, boolean> = $state({});

	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);
	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);
	const renderThinkingAsMarkdown = $derived(config().renderThinkingAsMarkdown as boolean);
	const showMessageStats = $derived(Boolean(config().showMessageStats));
	const showAgenticTurnStats = $derived(
		showMessageStats && Boolean(config().showAgenticTurnStats)
	);

	const hasReasoningError = $derived(
		isLastAssistantMessage ? !!agenticLastError(message.convId) : false
	);

	let permissionDismissed = $state(false);

	const pendingPermission = $derived(
		isStreaming && isLastAssistantMessage ? agenticPendingPermissionRequest(message.convId) : null
	);

	// Reset dismissed when pendingPermission changes (new request or cleared)
	let prevPendingRef: typeof pendingPermission = null;
	$effect(() => {
		if (pendingPermission !== prevPendingRef) {
			prevPendingRef = pendingPermission;
			if (pendingPermission) {
				permissionDismissed = false;
			}
		}
	});

	function handlePermission(decision: ToolPermissionDecision) {
		permissionDismissed = true;
		agenticResolvePermission(message.convId, decision);
	}

	let continueDismissed = $state(false);

	const pendingContinue = $derived(
		isStreaming && isLastAssistantMessage ? agenticPendingContinueRequest(message.convId) : false
	);

	let prevContinueRef = false;
	$effect(() => {
		if (pendingContinue !== prevContinueRef) {
			prevContinueRef = pendingContinue;
			if (pendingContinue) {
				continueDismissed = false;
			}
		}
	});

	function handleContinue(shouldContinue: boolean) {
		continueDismissed = true;
		agenticResolveContinue(message.convId, shouldContinue);
	}

	const sections = $derived(deriveAgenticSections(message, toolMessages, [], isStreaming));

	// Parse tool results with images
	const sectionsParsed = $derived(
		sections.map((section) => ({
			...section,
			parsedLines: section.toolResult
				? parseToolResultWithImages(section.toolResult, section.toolResultExtras || message?.extra)
				: ([] as ToolResultLine[])
		}))
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

	function buildTurnAgenticTimings(stats: ChatMessageAgenticTurnStats): ChatMessageAgenticTimings {
		return {
			turns: 1,
			toolCallsCount: stats.toolCalls.length,
			toolsMs: stats.toolsMs,
			toolCalls: stats.toolCalls,
			llm: stats.llm
		};
	}

	type ReadFileMeta = {
		fileName: string;
		lineRange: { start: number; end: number } | null;
		language: string;
	};

	function parseReadFileMeta(toolName: string | undefined, toolArgsString: string | undefined): ReadFileMeta | null {
		if (toolName !== 'read_file' || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(/[\\/]/).pop() || rawPath;

		const startRaw = args.start_line ?? args.line_start ?? args.startLine ?? args.from_line;
		const endRaw = args.end_line ?? args.line_end ?? args.endLine ?? args.to_line;
		const countRaw = args.line_count ?? args.count ?? args.num_lines;

		let lineRange: { start: number; end: number } | null = null;
		const sNum = Number(startRaw);
		const eNum = Number(endRaw);
		if (startRaw != null && endRaw != null && Number.isFinite(sNum) && Number.isFinite(eNum)) {
			lineRange = { start: sNum, end: eNum };
		} else if (startRaw != null && countRaw != null) {
			const cNum = Number(countRaw);
			if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
				lineRange = { start: sNum, end: sNum + cNum - 1 };
			}
		}

		const fileType = getFileTypeByExtension(fileName);
		const language = fileType ? fileType.replace(/^text:/, '') : 'text';

		return { fileName, lineRange, language };
	}
</script>

{#snippet renderSection(section: (typeof sectionsParsed)[number], index: number)}
	{#if section.type === AgenticSectionType.TEXT}
		<div class="agentic-text">
			<MarkdownContent content={section.content} attachments={message?.extra} />
		</div>
	{:else if section.type === AgenticSectionType.TOOL_CALL_STREAMING}
		{@const streamingIcon = isStreaming ? Loader2 : Loader2}
		{@const streamingIconClass = isStreaming ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={streamingIcon}
			iconClass={streamingIconClass}
			title={section.toolName || 'Tool call'}
			subtitle={isStreaming ? '' : 'incomplete'}
			{isStreaming}
			onToggle={() => toggleExpanded(index, section)}
		>
			<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
				<span>Input</span>

				{#if isStreaming}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>
			{#if section.toolArgs}
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight="22rem"
				/>
			{:else if isStreaming}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Receiving arguments...
				</div>
			{:else}
				<div
					class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
				>
					Response was truncated
				</div>
			{/if}
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING}
		{@const isPending = section.type === AgenticSectionType.TOOL_CALL_PENDING}
		{@const toolIcon = isPending ? Loader2 : Wrench}
		{@const toolIconClass = isPending ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}
		{@const readFileMeta = parseReadFileMeta(section.toolName, section.toolArgs)}

		{#snippet readFileTitle()}
			<span class="text-muted-foreground">Read file </span>
			<span class="font-mono">{readFileMeta?.fileName}</span>
			{#if readFileMeta?.lineRange}
				<span class="text-muted-foreground"
					>{' '}(lines {readFileMeta.lineRange.start}-{readFileMeta.lineRange.end})</span
				>
			{/if}
		{/snippet}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={toolIcon}
			iconClass={toolIconClass}
			title={readFileMeta ? '' : section.toolName || ''}
			titleSnippet={readFileMeta ? readFileTitle : undefined}
			subtitle={isPending ? 'executing...' : undefined}
			isStreaming={isPending}
			onToggle={() => toggleExpanded(index, section)}
		>
			{#if section.toolArgs && section.toolArgs !== '{}'}
				<div class="mb-1.5 text-xs text-muted-foreground/70">Input</div>

				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight="22rem"
				/>
			{/if}

			<div class={section.toolArgs && section.toolArgs !== '{}' ? 'mt-4' : ''}>
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					<span>Output</span>

					{#if isPending}
						<Loader2 class="h-3 w-3 animate-spin" />
					{/if}
				</div>
				{#if isPending}
					<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
						Waiting for result...
					</div>
				{:else if section.toolResult}
					{#if readFileMeta}
						<SyntaxHighlightedCode
							code={section.toolResult}
							language={readFileMeta.language}
							maxHeight="22rem"
						/>
					{:else}
						<div class="overflow-auto">
							{#each section.parsedLines as line, i (i)}
								<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
									{line.text}
								</div>
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
					{/if}
				{:else}
					<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
				{/if}
			</div>
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.REASONING}
		{@const reasoningSubtitle = section.wasInterrupted
			? hasReasoningError
				? 'Error'
				: 'Cancelled'
			: isStreaming
				? ''
				: undefined}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={Lightbulb}
			iconClass="h-3.5 w-3.5"
			title="Reasoning"
			subtitle={reasoningSubtitle}
			rawContent={section.content}
			onToggle={() => toggleExpanded(index, section)}
		>
			{#if renderThinkingAsMarkdown}
				<MarkdownContent content={section.content} attachments={message?.extra} />
			{:else}
				<div class="text-[13px] leading-relaxed break-words whitespace-pre-wrap text-foreground/90">
					{section.content}
				</div>
			{/if}
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.REASONING_PENDING}
		{@const reasoningTitle = isStreaming ? 'Reasoning...' : 'Reasoning'}
		{@const reasoningSubtitle = isStreaming ? '' : hasReasoningError ? 'Error' : 'Cancelled'}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={Lightbulb}
			iconClass="h-3.5 w-3.5"
			title={reasoningTitle}
			subtitle={reasoningSubtitle}
			rawContent={section.content}
			{isStreaming}
			shimmerTitle={isStreaming}
			onToggle={() => toggleExpanded(index, section)}
		>
			{#if renderThinkingAsMarkdown}
				<MarkdownContent content={section.content} attachments={message?.extra} />
			{:else}
				<div class="text-[13px] leading-relaxed break-words whitespace-pre-wrap text-foreground/90">
					{section.content}
				</div>
			{/if}
		</CollapsibleContentBlock>
	{/if}
{/snippet}

<div class="agentic-content">
	{#if turnGroups.length > 1}
		{#each turnGroups as turn, turnIndex (turnIndex)}
			{@const turnStats = message?.timings?.agentic?.perTurn?.[turnIndex]}

			<div class="agentic-turn group/turn grid gap-3 mb-2">
				{#each turn.sections as section, sIdx (turn.flatIndices[sIdx])}
					{@render renderSection(section, turn.flatIndices[sIdx])}
				{/each}

				{#if turnStats && showAgenticTurnStats}
					<div class="turn-stats transition-opacity duration-150">
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
		{#each sectionsParsed as section, index (index)}
			{@render renderSection(section, index)}
		{/each}
	{/if}

	{#if pendingPermission && !permissionDismissed}
		<ChatMessageActionCardPermissionRequest
			toolName={pendingPermission.toolName}
			serverLabel={pendingPermission.serverLabel}
			onDecision={handlePermission}
		/>
	{/if}

	{#if pendingContinue && !continueDismissed}
		<ChatMessageActionCardContinueRequest onDecision={handleContinue} />
	{/if}
</div>

<style>
	.agentic-content {
		display: flex;
		flex-direction: column;
		width: 100%;
		max-width: 48rem;
		/*gap: 1rem;*/
	}

	.agentic-content > :global(*),
	.agentic-turn > :global(*) {
		min-width: 0;
	}

	.agentic-text {
		width: 100%;
	}

	.turn-stats {
		border-top: 1px solid hsl(var(--muted) / 0.5);
	}
</style>
