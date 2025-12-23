<script lang="ts">
	import {
		ModelBadge,
		ChatMessageActions,
		ChatMessageStatistics,
		ChatMessageThinkingBlock,
		MarkdownContent,
		ModelsSelector,
		BadgeChatStatistic
	} from '$lib/components/app';
	import type { DatabaseMessage, ApiChatCompletionToolCall } from '$lib/types';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { autoResizeTextarea, copyToClipboard } from '$lib/utils';
	import { fade } from 'svelte/transition';
	import { Check, Clock, X, Wrench } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import Label from '$lib/components/ui/label/label.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { SvelteSet } from 'svelte/reactivity';

	type ToolSegment =
		| { kind: 'content'; content: string; parentId: string }
		| { kind: 'thinking'; content: string }
		| {
				kind: 'tool';
				toolCalls: ApiChatCompletionToolCall[];
				parentId: string;
				inThinking: boolean;
		  };
	type ToolParsed = { expression?: string; result?: string; duration_ms?: number };
	type CollectedToolMessage = {
		toolCallId?: string | null;
		parsed: ToolParsed;
	};
	type MessageWithToolExtras = DatabaseMessage & {
		_segments?: ToolSegment[];
		_toolMessagesCollected?: CollectedToolMessage[];
	};
	type ToolMessageLike = Pick<DatabaseMessage, 'role' | 'content'> & {
		toolCallId?: string | null;
		parent?: string;
	};

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		editedContent?: string;
		isEditing?: boolean;
		message: DatabaseMessage;
		messageContent: string | undefined;
		onCancelEdit?: () => void;
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onEditKeydown?: (event: KeyboardEvent) => void;
		onEditedContentChange?: (content: string) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onSaveEdit?: () => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		onShouldBranchAfterEditChange?: (value: boolean) => void;
		showDeleteDialog: boolean;
		shouldBranchAfterEdit?: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
		thinkingContent: string | null;
		toolCallContent: ApiChatCompletionToolCall[] | string | null;
		toolParentIds?: string[];
		segments?: ToolSegment[] | null;
		toolMessagesCollected?: CollectedToolMessage[] | null;
	}

	let {
		class: className = '',
		deletionInfo,
		editedContent = '',
		isEditing = false,
		message,
		messageContent,
		onCancelEdit,
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onEditKeydown,
		onEditedContentChange,
		onNavigateToSibling,
		onRegenerate,
		onSaveEdit,
		onShowDeleteDialogChange,
		onShouldBranchAfterEditChange,
		showDeleteDialog,
		shouldBranchAfterEdit = false,
		siblingInfo = null,
		textareaElement = $bindable(),
		thinkingContent,
		toolCallContent = null,
		toolParentIds = [message.id],
		segments: segmentsProp = null,
		toolMessagesCollected: toolMessagesCollectedProp = (message as MessageWithToolExtras)
			._toolMessagesCollected ?? null
	}: Props = $props();

	// Keep segments/tool messages in sync with the merged assistant produced upstream.
	let segments = $derived(segmentsProp ?? (message as MessageWithToolExtras)._segments ?? null);
	let toolMessagesCollected = $derived(
		toolMessagesCollectedProp ?? (message as MessageWithToolExtras)._toolMessagesCollected ?? null
	);

	let hasRegularContent = $derived.by(() => {
		if (messageContent?.trim()) return true;
		return (segments ?? []).some((s) => s.kind === 'content' && Boolean(s.content?.trim()));
	});

	const toolCalls = $derived(
		Array.isArray(toolCallContent) ? (toolCallContent as ApiChatCompletionToolCall[]) : null
	);

	const processingState = useProcessingState();
	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	const toolMessages = $derived<ToolMessageLike[]>(
		(() => {
			const ids = new SvelteSet<string>();
			if (toolCalls) {
				for (const tc of toolCalls) {
					if (tc.id) ids.add(tc.id);
				}
			}
			const collected = toolMessagesCollected ?? [];
			return conversationsStore.activeMessages
				.filter(
					(m) =>
						m.role === 'tool' &&
						(toolParentIds.includes(m.parent) || (m.toolCallId && ids.has(m.toolCallId)))
				)
				.concat(
					collected.map((c) => ({
						role: 'tool',
						content: JSON.stringify(c.parsed),
						toolCallId: c.toolCallId ?? undefined,
						parent: toolParentIds[0]
					}))
				);
		})()
	);
	const toolMessagesById = $derived<Record<string, ToolParsed>>(
		(() => {
			const map: Record<string, ToolParsed> = {};
			for (const t of toolMessages) {
				const parsed = parseToolMessage(t);
				if (parsed && t.toolCallId) {
					map[t.toolCallId] = parsed;
				}
			}
			return map;
		})()
	);

	const collectedById = $derived<Record<string, ToolParsed>>(
		(() => {
			const map: Record<string, ToolParsed> = {};
			(toolMessagesCollected ?? []).forEach((c) => {
				if (c.toolCallId) {
					map[c.toolCallId] = c.parsed;
				}
			});
			return map;
		})()
	);

	function getToolResult(toolCall: ApiChatCompletionToolCall): ToolParsed | null {
		const idSetMatch = toolCall.id ? toolMessagesById[toolCall.id] : null;
		if (idSetMatch) return idSetMatch;
		if (toolCall.id && collectedById[toolCall.id]) return collectedById[toolCall.id];
		return null;
	}

	function advanceToolResult(toolCall: ApiChatCompletionToolCall) {
		return getToolResult(toolCall) ?? null;
	}
	let displayedModel = $derived((): string | null => {
		if (message.model) {
			return message.model;
		}

		return null;
	});

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities: () => conversationsStore.getModalitiesUpToMessage(message.id),
		onSuccess: (modelName) => onRegenerate(modelName)
	});

	function handleCopyModel() {
		const model = displayedModel();

		void copyToClipboard(model ?? '');
	}

	$effect(() => {
		if (isEditing && textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});

	function parseArguments(
		toolCall: ApiChatCompletionToolCall
	): { pairs: { key: string; value: string }[] } | { raw: string } | null {
		const rawArguments = toolCall.function?.arguments?.trim();
		if (!rawArguments) return null;
		try {
			const parsed = JSON.parse(rawArguments);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				const pairs = Object.entries(parsed).map(([key, value]) => ({
					key,
					value: typeof value === 'string' ? value : JSON.stringify(value, null, 2)
				}));
				return { pairs };
			}
		} catch {
			// ignore parse errors, fall back to raw
		}
		return { raw: rawArguments };
	}

	function parseToolMessage(msg: ToolMessageLike): ToolParsed | null {
		if (!msg.content) return null;
		try {
			const parsed = JSON.parse(msg.content);
			if (parsed && typeof parsed === 'object') {
				const duration =
					typeof parsed.duration_ms === 'number' ? (parsed.duration_ms as number) : undefined;
				return {
					expression: parsed.expression ?? undefined,
					result: parsed.result ?? undefined,
					duration_ms: duration
				};
			}
		} catch {
			// not JSON; fall back
		}
		return { result: msg.content };
	}

	function formatDurationSeconds(durationMs?: number): string | null {
		if (durationMs === undefined) return null;
		if (!Number.isFinite(durationMs)) return null;
		return `${(durationMs / 1000).toFixed(2)}s`;
	}

	function toFencedCodeBlock(code: string, language: string): string {
		const matches = code.match(/`+/g) ?? [];
		const maxBackticks = matches.reduce((max, s) => Math.max(max, s.length), 0);
		const fence = '`'.repeat(Math.max(3, maxBackticks + 1));
		return `${fence}${language}\n${code}\n${fence}`;
	}

	function getToolLabel(toolCall: ApiChatCompletionToolCall, index: number) {
		const name = toolCall.function?.name ?? '';
		if (name === 'calculator') return 'Calculator';
		if (name === 'code_interpreter_javascript') return 'Code Interpreter (JavaScript)';
		return name || `Call #${index + 1}`;
	}

	function segmentToolInThinking(segment: ToolSegment): boolean {
		if (segment.kind !== 'tool') return false;
		const maybe = segment as unknown as { inThinking?: unknown };
		if (typeof maybe.inThinking === 'boolean') return maybe.inThinking;
		// Back-compat fallback: if we don't know, treat as in-reasoning when there is a thinking block.
		return Boolean(thinkingContent);
	}
</script>

<div
	class="text-md group w-full leading-7.5 {className}"
	role="group"
	aria-label="Assistant message with actions"
>
	{#if thinkingContent}
		<ChatMessageThinkingBlock
			reasoningContent={segments && segments.length ? null : thinkingContent}
			isStreaming={!message.timestamp || isLoading()}
			{hasRegularContent}
		>
			{#if segments && segments.length}
				{#each segments as segment, segIndex (segIndex)}
					{#if segment.kind === 'thinking'}
						<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
							{segment.content}
						</div>
					{:else if segment.kind === 'tool' && segmentToolInThinking(segment)}
						{#each segment.toolCalls as toolCall, index (toolCall.id ?? `${segIndex}-${index}`)}
							{@const argsParsed = parseArguments(toolCall)}
							{@const parsed = advanceToolResult(toolCall)}
							{@const collectedResult = toolMessagesCollected
								? toolMessagesCollected.find((c) => c.toolCallId === toolCall.id)?.parsed?.result
								: undefined}
							{@const collectedDurationMs = toolMessagesCollected
								? toolMessagesCollected.find((c) => c.toolCallId === toolCall.id)?.parsed
										?.duration_ms
								: undefined}
							{@const durationMs = parsed?.duration_ms ?? collectedDurationMs}
							{@const durationText = formatDurationSeconds(durationMs)}
							<div
								class="mt-2 space-y-1 rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-2.5 py-2"
								data-testid="tool-call-block"
							>
								<div class="flex items-center justify-between gap-2">
									<div class="flex items-center gap-1 text-xs font-semibold">
										<Wrench class="h-3.5 w-3.5" />
										<span>{getToolLabel(toolCall, index)}</span>
									</div>
									{#if durationText}
										<BadgeChatStatistic icon={Clock} value={durationText} />
									{/if}
								</div>
								{#if argsParsed}
									<div class="text-[12px] text-muted-foreground">Arguments</div>
									{#if 'pairs' in argsParsed}
										{#each argsParsed.pairs as pair (pair.key)}
											<div class="mt-1 rounded-sm bg-background/70 px-2 py-1.5">
												<div class="text-[12px] font-semibold text-foreground">{pair.key}</div>
												{#if pair.key === 'code' && toolCall.function?.name === 'code_interpreter_javascript'}
													<MarkdownContent
														class="mt-0.5 text-[12px] leading-snug"
														content={toFencedCodeBlock(pair.value, 'javascript')}
													/>
												{:else}
													<pre
														class="mt-0.5 font-mono text-[12px] leading-snug break-words whitespace-pre-wrap">
{pair.value}
													</pre>
												{/if}
											</div>
										{/each}
									{:else}
										<pre class="font-mono text-[12px] leading-snug break-words whitespace-pre-wrap">
{argsParsed.raw}
										</pre>
									{/if}
								{/if}
								{#if parsed && parsed.result !== undefined}
									<div class="text-[12px] text-muted-foreground">Result</div>
									<div class="rounded-sm bg-background/80 px-2 py-1 font-mono text-[12px]">
										{parsed.result}
									</div>
								{:else if collectedResult !== undefined}
									<div class="text-[12px] text-muted-foreground">Result</div>
									<div class="rounded-sm bg-background/80 px-2 py-1 font-mono text-[12px]">
										{collectedResult}
									</div>
								{/if}
							</div>
						{/each}
					{/if}
				{/each}
			{/if}
		</ChatMessageThinkingBlock>
	{/if}

	{#if message?.role === 'assistant' && isLoading() && !message?.content?.trim()}
		<div class="mt-6 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getProcessingMessage()}
				</span>
			</div>
		</div>
	{/if}

	{#if isEditing}
		<div class="w-full">
			<textarea
				bind:this={textareaElement}
				bind:value={editedContent}
				class="min-h-[50vh] w-full resize-y rounded-2xl px-3 py-2 text-sm {INPUT_CLASSES}"
				onkeydown={onEditKeydown}
				oninput={(e) => {
					autoResizeTextarea(e.currentTarget);
					onEditedContentChange?.(e.currentTarget.value);
				}}
				placeholder="Edit assistant message..."
			></textarea>

			<div class="mt-2 flex items-center justify-between">
				<div class="flex items-center space-x-2">
					<Checkbox
						id="branch-after-edit"
						bind:checked={shouldBranchAfterEdit}
						onCheckedChange={(checked) => onShouldBranchAfterEditChange?.(checked === true)}
					/>
					<Label for="branch-after-edit" class="cursor-pointer text-sm text-muted-foreground">
						Branch conversation after edit
					</Label>
				</div>
				<div class="flex gap-2">
					<Button class="h-8 px-3" onclick={onCancelEdit} size="sm" variant="outline">
						<X class="mr-1 h-3 w-3" />
						Cancel
					</Button>

					<Button class="h-8 px-3" onclick={onSaveEdit} disabled={!editedContent?.trim()} size="sm">
						<Check class="mr-1 h-3 w-3" />
						Save
					</Button>
				</div>
			</div>
		</div>
	{:else if message.role === 'assistant'}
		{#if config().disableReasoningFormat}
			<pre class="raw-output">{messageContent}</pre>
		{:else if segments && segments.length}
			{#each segments as segment, segIndex (segIndex)}
				{#if segment.kind === 'content'}
					<MarkdownContent content={segment.content ?? ''} />
				{:else if segment.kind === 'tool' && (!thinkingContent || !segmentToolInThinking(segment))}
					{#each segment.toolCalls as toolCall, index (toolCall.id ?? `${segIndex}-${index}`)}
						{@const argsParsed = parseArguments(toolCall)}
						{@const parsed = advanceToolResult(toolCall)}
						{@const collectedResult = toolMessagesCollected
							? toolMessagesCollected.find((c) => c.toolCallId === toolCall.id)?.parsed?.result
							: undefined}
						{@const collectedDurationMs = toolMessagesCollected
							? toolMessagesCollected.find((c) => c.toolCallId === toolCall.id)?.parsed?.duration_ms
							: undefined}
						{@const durationMs = parsed?.duration_ms ?? collectedDurationMs}
						{@const durationText = formatDurationSeconds(durationMs)}
						<div
							class="mt-2 space-y-1 rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-2.5 py-2"
							data-testid="tool-call-block"
						>
							<div class="flex items-center justify-between gap-2">
								<div class="flex items-center gap-1 text-xs font-semibold">
									<Wrench class="h-3.5 w-3.5" />
									<span>{getToolLabel(toolCall, index)}</span>
								</div>
								{#if durationText}
									<BadgeChatStatistic icon={Clock} value={durationText} />
								{/if}
							</div>
							{#if argsParsed}
								<div class="text-[12px] text-muted-foreground">Arguments</div>
								{#if 'pairs' in argsParsed}
									{#each argsParsed.pairs as pair (pair.key)}
										<div class="mt-1 rounded-sm bg-background/70 px-2 py-1.5">
											<div class="text-[12px] font-semibold text-foreground">{pair.key}</div>
											{#if pair.key === 'code' && toolCall.function?.name === 'code_interpreter_javascript'}
												<MarkdownContent
													class="mt-0.5 text-[12px] leading-snug"
													content={toFencedCodeBlock(pair.value, 'javascript')}
												/>
											{:else}
												<pre
													class="mt-0.5 font-mono text-[12px] leading-snug break-words whitespace-pre-wrap">
{pair.value}
												</pre>
											{/if}
										</div>
									{/each}
								{:else}
									<pre class="font-mono text-[12px] leading-snug break-words whitespace-pre-wrap">
{argsParsed.raw}
									</pre>
								{/if}
							{/if}
							{#if parsed && parsed.result !== undefined}
								<div class="text-[12px] text-muted-foreground">Result</div>
								<div class="rounded-sm bg-background/80 px-2 py-1 font-mono text-[12px]">
									{parsed.result}
								</div>
							{:else if collectedResult !== undefined}
								<div class="text-[12px] text-muted-foreground">Result</div>
								<div class="rounded-sm bg-background/80 px-2 py-1 font-mono text-[12px]">
									{collectedResult}
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			{/each}
		{:else}
			<MarkdownContent content={messageContent ?? ''} />
		{/if}
	{:else}
		<div class="text-sm whitespace-pre-wrap">
			{messageContent}
		</div>
	{/if}

	<div class="info my-6 grid gap-4">
		{#if displayedModel()}
			<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
				{#if isRouter}
					<ModelsSelector
						currentModel={displayedModel()}
						onModelChange={handleModelChange}
						disabled={isLoading()}
						upToMessageId={message.id}
					/>
				{:else}
					<ModelBadge model={displayedModel() || undefined} onclick={handleCopyModel} />
				{/if}

				{#if currentConfig.showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
					<ChatMessageStatistics
						promptTokens={message.timings.prompt_n}
						promptMs={message.timings.prompt_ms}
						predictedTokens={message.timings.predicted_n}
						predictedMs={message.timings.predicted_ms}
					/>
				{/if}
			</div>
		{/if}
	</div>

	{#if message.timestamp && !isEditing}
		<ChatMessageActions
			role="assistant"
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration && !thinkingContent
				? onContinue
				: undefined}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
		/>
	{/if}
</div>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}

	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.tool-call-badge {
		max-width: 12rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.tool-call-badge--fallback {
		max-width: 20rem;
		white-space: normal;
		word-break: break-word;
	}
</style>
