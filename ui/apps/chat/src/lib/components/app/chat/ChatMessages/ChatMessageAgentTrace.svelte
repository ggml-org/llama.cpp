<script lang="ts">
	import ChatMessageToolCallBlock from './ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
	import ChatMessageToolCallGroup from './ChatMessage/ChatMessageToolCall/ChatMessageToolCallGroup.svelte';
	import { parseExecShellCommandMeta } from './ChatMessage/ChatMessageToolCall/parsers/exec-shell-command';
	import { setToolCallElapsedContext } from './ChatMessage/ChatMessageToolCall/tool-call-elapsed';
	import ChatMessageReasoningBlock from './ChatMessageReasoningBlock.svelte';
	import { groupToolCallRuns, isToolCallStep } from '@ggml-org/llama-harness';
	import {
		ChatMessageActionCardContinueRequest,
		ChatMessageActionCardPermissionRequest,
		ChatMessageAssistantProcessingInfo,
		ChatMessageStatistics,
		MarkdownContent
	} from '#lib/components/app/index.ts';
	import { REPLY_STATUS_FALLBACK, TOOL_CALL_GROUP_MIN_CALLS } from '#lib/constants/index.ts';
	import {
		AgenticSectionType,
		ChatMessageStatsView,
		ToolPermissionDecision
	} from '#lib/enums/index.ts';
	import { agenticStore, conversationsStore, settingsStore } from '#lib/stores/index.ts';
	import type {
		AgenticSection,
		ChatMessageAgenticTimings,
		ChatMessageAgenticTurnStats,
		DatabaseMessage
	} from '#lib/types/index.ts';
	import {
		deriveAgenticSections,
		isSectionExpandedByDefault,
		showsReplyStatus
	} from '#lib/utils/index.ts';

	interface Props {
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
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
		statusText = null,
		toolMessages = []
	}: Props = $props();

	// The chat-level streaming flag is shared by every message; only the last
	// assistant message is actually live, so earlier ones must not keep their
	// unresolved tool calls spinning while a newer reply streams.
	const isStreaming = $derived(isChatStreaming && isLastAssistantMessage);

	let expandedStates: Record<number, boolean> = $state({});
	// keyed by the flat index of the group's first call
	let groupExpandedStates: Record<number, boolean> = $state({});

	const showThoughtInProgress = $derived(Boolean(settingsStore.config.showThoughtInProgress));
	const showToolCallInProgress = $derived(Boolean(settingsStore.config.showToolCallInProgress));
	const alwaysShowToolCallContent = $derived(
		Boolean(settingsStore.config.alwaysShowToolCallContent)
	);
	const showMessageStats = $derived(Boolean(settingsStore.config.showMessageStats));
	const showAgenticTurnStats = $derived(
		showMessageStats && Boolean(settingsStore.config.showAgenticTurnStats)
	);

	const hasReasoningError = $derived(
		isLastAssistantMessage ? !!agenticStore.getLastError(message.convId) : false
	);

	const pendingPermission = $derived(
		isStreaming ? agenticStore.getPendingPermissionRequest(message.convId) : null
	);

	// dismissal applies to the request object, so the next request ( new
	// identity ) shows the card again without any reset bookkeeping
	let dismissedPermission: typeof pendingPermission = $state(null);

	const visiblePermission = $derived(
		pendingPermission && dismissedPermission !== pendingPermission ? pendingPermission : null
	);

	function handlePermission(decision: ToolPermissionDecision) {
		dismissedPermission = pendingPermission;
		agenticStore.resolvePermission(message.convId, decision);
	}

	const pendingContinue = $derived(
		isStreaming ? agenticStore.getPendingContinueRequest(message.convId) : false
	);

	let continueDismissed = $state(false);

	// the continue request is a plain boolean, so there is no identity to
	// compare against; clear the dismissal whenever no request is pending so
	// the next one starts from a clean state
	$effect(() => {
		if (!pendingContinue) {
			continueDismissed = false;
		}
	});

	const showContinue = $derived(Boolean(pendingContinue) && !continueDismissed);

	function handleContinue(shouldContinue: boolean) {
		continueDismissed = true;
		agenticStore.resolveContinue(message.convId, shouldContinue);
	}

	const sections = $derived(deriveAgenticSections(message, toolMessages, [], isStreaming));

	// The command the pending call would run, for the permission card: the gate
	// names the call it is asking about, so an earlier call of the same tool
	// whose result has not landed yet cannot be mistaken for it.
	const pendingCommand = $derived.by(() => {
		if (!visiblePermission) return undefined;

		const pending = sections.find((section) => section.toolCallId === visiblePermission.toolCallId);

		return pending ? parseExecShellCommandMeta(pending)?.command : undefined;
	});

	const currentlyExecutingToolCallId = $derived(
		isStreaming ? agenticStore.getExecutingToolCallId(message.convId) : null
	);

	// Cwd the agentic loop passes to tool calls in this conversation; the exec
	// block shows it until the result message persists the real one.
	const conversationCwd = $derived(
		conversationsStore.activeConversation?.id === message.convId
			? conversationsStore.activeConversation?.cwd
			: undefined
	);

	/** A single section by its index, or a run of them shown as one group row. */
	type TraceItem = number | number[];

	// The rows of the reply, as one list per turn. Per-turn stats sit between
	// turns, so with them on each turn groups its own calls. Without them a
	// run of calls spans turns ( a model that thinks before each call ) and
	// the reply is a single list.
	const turns: TraceItem[][] = $derived.by(() => {
		if (!showAgenticTurnStats) {
			return [
				groupToolCallRuns(
					sections,
					sections.map((_, i) => i),
					TOOL_CALL_GROUP_MIN_CALLS
				)
			];
		}

		const result: TraceItem[][] = [];

		let indices: number[] = [];
		let prevWasTool = false;

		const pushTurn = () =>
			result.push(groupToolCallRuns(sections, indices, TOOL_CALL_GROUP_MIN_CALLS));

		for (let i = 0; i < sections.length; i++) {
			const isTool = isToolCallStep(sections[i]);

			// a turn ends where its tool calls do
			if (!isTool && prevWasTool && indices.length > 0) {
				pushTurn();
				indices = [];
			}

			indices.push(i);
			prevWasTool = isTool;
		}

		if (indices.length > 0) pushTurn();

		return result;
	});

	// a single turn has no stats of its own to show
	const showTurnStats = $derived(showAgenticTurnStats && turns.length > 1);

	const lastItem = $derived(turns.at(-1)?.at(-1));

	// Each tool row times its own loop of calls, from its first call's start
	// to its last call's end; the run stamps those times on the root message's
	// timings, and a call still running reads its start from the live session.
	const callTimings = $derived(
		new Map(
			(message?.timings?.agentic?.toolCalls ?? []).flatMap((timing) =>
				timing.id ? [[timing.id, timing] as const] : []
			)
		)
	);

	function callStartedAt(toolCallId: string | undefined): number | null {
		if (!toolCallId) return null;

		return (
			callTimings.get(toolCallId)?.startedAt ??
			(isStreaming ? agenticStore.getToolCallStartedAt(message.convId, toolCallId) : null)
		);
	}

	/** The loop of a group: its start, and its end once done ( null while it runs ). */
	function loopTiming(item: number[]): { startedAt: number; endedAt: number | null } | null {
		// a group can open with the reasoning that led to its first call
		const ids = item.flatMap((index) => sections[index]?.toolCallId ?? []);
		const startedAt = callStartedAt(ids[0]);

		if (startedAt === null) return null;

		// the last row of a live reply is the loop still running
		if (isStreaming && item === lastItem) return { endedAt: null, startedAt };

		const ends = ids.map((id) => callTimings.get(id)?.endedAt);

		return ends.every((end) => end !== undefined)
			? { endedAt: Math.max(...(ends as number[])), startedAt }
			: null;
	}

	// a single call closing a live reply reads its timer from context by its id
	setToolCallElapsedContext(() => {
		if (!isStreaming || lastItem === undefined || Array.isArray(lastItem)) return null;

		const toolCallId = sections[lastItem]?.toolCallId;
		const startedAt = callStartedAt(toolCallId);

		return toolCallId && startedAt !== null ? { startedAt, toolCallId } : null;
	});

	const lastToolIndex = $derived(sections.findLastIndex((section) => isToolCallStep(section)));

	// The status row shows only while nothing else in the reply is moving: a
	// running row, a permission card or streaming text already says what is
	// happening. It sits where the next row will land, so that row replaces
	// it and the stats below stay put.
	const showStatus = $derived(
		currentlyExecutingToolCallId === null &&
			!visiblePermission &&
			!showContinue &&
			showsReplyStatus(statusText, sections.at(-1), isStreaming)
	);

	// a group closing the reply carries the status as its live title, so the
	// waiting reads as part of the work instead of a line of its own
	const statusInGroup = $derived(showStatus && Array.isArray(lastItem));

	function isExecutingCall(section: AgenticSection): boolean {
		return section.toolCallId !== undefined && section.toolCallId === currentlyExecutingToolCallId;
	}

	function isExpanded(index: number, section: AgenticSection, inOpenGroup = false): boolean {
		if (expandedStates[index] !== undefined) {
			return expandedStates[index];
		}

		// opening a group while a call runs shows that call's live output
		if (inOpenGroup && isExecutingCall(section)) return true;

		return isSectionExpandedByDefault(section, {
			alwaysShowToolCallContent,
			executingToolCallId: currentlyExecutingToolCallId,
			showThoughtInProgress,
			showToolCallInProgress
		});
	}

	function toggleExpanded(index: number, section: AgenticSection, inOpenGroup = false) {
		const currentState = isExpanded(index, section, inOpenGroup);

		expandedStates[index] = !currentState;
	}

	function isGroupExpanded(indices: number[]): boolean {
		const state = groupExpandedStates[indices[0]];

		if (state !== undefined) return state;

		if (alwaysShowToolCallContent) return true;

		// with "Show tool call in progress" the latest group stays open for the
		// whole reply, so calls are seen as they happen, and folds into its
		// summary once the reply is done
		return showToolCallInProgress && isStreaming && indices.includes(lastToolIndex);
	}

	function toggleGroupExpanded(indices: number[]) {
		groupExpandedStates[indices[0]] = !isGroupExpanded(indices);
	}

	function buildTurnAgenticTimings(stats: ChatMessageAgenticTurnStats): ChatMessageAgenticTimings {
		return {
			llm: stats.llm,
			toolCalls: stats.toolCalls,
			toolCallsCount: stats.toolCalls.length,
			toolsMs: stats.toolsMs,
			turns: 1
		};
	}
</script>

{#snippet renderSection(section: AgenticSection, index: number, inOpenGroup = false)}
	{#if section.type === AgenticSectionType.TEXT}
		<div class="agent-trace-text">
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
	{:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING || section.type === AgenticSectionType.TOOL_CALL_STREAMING}
		<ChatMessageToolCallBlock
			attachments={section.toolResultExtras}
			{conversationCwd}
			isExecuting={isExecutingCall(section)}
			{isStreaming}
			onToggle={() => toggleExpanded(index, section, inOpenGroup)}
			open={isExpanded(index, section, inOpenGroup)}
			{section}
		/>
	{/if}
{/snippet}

{#snippet renderItem(item: TraceItem)}
	{#if Array.isArray(item)}
		{@const groupOpen = isGroupExpanded(item)}
		{@const loop = loopTiming(item)}
		<ChatMessageToolCallGroup
			elapsedEndedAt={loop?.endedAt ?? null}
			elapsedStartedAt={loop?.startedAt ?? null}
			executingToolCallId={currentlyExecutingToolCallId}
			{isStreaming}
			onToggle={() => toggleGroupExpanded(item)}
			open={groupOpen}
			sections={item.map((i) => sections[i])}
			status={statusInGroup && item === lastItem ? statusText || REPLY_STATUS_FALLBACK : null}
		>
			{#each item as index (index)}
				{@render renderSection(sections[index], index, groupOpen)}
			{/each}
		</ChatMessageToolCallGroup>
	{:else}
		{@render renderSection(sections[item], item)}
	{/if}
{/snippet}

{#snippet renderItems(items: TraceItem[])}
	{#each items as item (Array.isArray(item) ? `g${item[0]}` : item)}
		{@render renderItem(item)}
	{/each}
{/snippet}

<div class="agent-trace gap-2">
	{#each turns as items, turnIndex (turnIndex)}
		{#if showTurnStats}
			{@const turnStats = message?.timings?.agentic?.perTurn?.[turnIndex]}

			<div class="agent-trace-turn group/turn grid gap-2">
				{@render renderItems(items)}

				{#if turnStats}
					<div class="turn-stats transition-opacity duration-150 mt-1 mb-4">
						<ChatMessageStatistics
							agenticTimings={turnStats.toolCalls.length > 0
								? buildTurnAgenticTimings(turnStats)
								: undefined}
							hideSummary
							initialView={ChatMessageStatsView.GENERATION}
							predictedMs={turnStats.llm.predicted_ms}
							predictedTokens={turnStats.llm.predicted_n}
							promptMs={turnStats.llm.prompt_ms}
							promptTokens={turnStats.llm.prompt_n}
						/>
					</div>
				{/if}
			</div>
		{:else}
			{@render renderItems(items)}
		{/if}
	{/each}

	{#if visiblePermission}
		<ChatMessageActionCardPermissionRequest
			command={pendingCommand}
			onDecision={handlePermission}
			serverLabel={visiblePermission.serverLabel}
			toolName={visiblePermission.toolName}
		/>
	{/if}

	{#if showContinue}
		<ChatMessageActionCardContinueRequest onDecision={handleContinue} />
	{/if}

	{#if showStatus && !statusInGroup}
		<ChatMessageAssistantProcessingInfo text={statusText || REPLY_STATUS_FALLBACK} />
	{/if}
</div>

<style>
	.agent-trace {
		display: flex;
		flex-direction: column;
		width: 100%;
		max-width: 48rem;
	}

	.agent-trace > :global(*),
	.agent-trace-turn > :global(*) {
		min-width: 0;
	}

	.agent-trace-text {
		width: 100%;
	}

	/* the answer stands a little apart from the work around it: trace rows keep
	   their tight 0.5rem stack, text and a trace row get 0.75rem between them */
	.agent-trace > :global(:not(.agent-trace-text) + .agent-trace-text),
	.agent-trace-turn > :global(:not(.agent-trace-text) + .agent-trace-text) {
		margin-top: 0.25rem;
	}

	/* on the text, not the row: the collapsible rows pin their own margins */
	.agent-trace > :global(.agent-trace-text:has(+ :not(.agent-trace-text))),
	.agent-trace-turn > :global(.agent-trace-text:has(+ :not(.agent-trace-text))) {
		margin-bottom: 0.25rem;
	}

	.turn-stats {
		border-top: 1px solid hsl(var(--surface-muted) / 0.5);
	}
</style>
