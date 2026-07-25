<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageStatistics,
		ChatMessageUserBubble
	} from '$lib/components/app/chat';
	import { getMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { MessageRole, ChatMessageStatisticsMode } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		siblingInfo?: ChatMessageSiblingInfo | null;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		isLastUserMessage?: boolean;
		nextAssistantMessage?: DatabaseMessage | null;
		showDeleteDialog: boolean;
		onEdit: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onCopy: () => void;
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		deletionInfo,
		isLastUserMessage = false,
		nextAssistantMessage = null,
		showDeleteDialog,
		onEdit,
		onDelete,
		onConfirmDelete,
		onForkConversation,
		onShowDeleteDialogChange,
		onNavigateToSibling,
		onCopy
	}: Props = $props();

	// Get contexts
	const editCtx = getMessageEditContext();
	const processingState = useProcessingState();

	const currentConfig = $derived(config());
	const isActivelyProcessing = $derived(isLastUserMessage && isLoading());

	// For agentic turns, prefer the cumulative agentic.llm totals over per-call timings.
	let storedReadingStats = $derived.by(() => {
		const timings = nextAssistantMessage?.timings;
		if (!timings?.prompt_n || !timings?.prompt_ms) return null;

		const agentic = timings.agentic;

		return {
			promptTokens: agentic ? agentic.llm.prompt_n : timings.prompt_n,
			promptMs: agentic ? agentic.llm.prompt_ms : timings.prompt_ms
		};
	});

	let showStoredReadingStats = $derived(
		Boolean(currentConfig.showMessageStats) && storedReadingStats !== null
	);

	let showLiveReadingStats = $derived(
		Boolean(currentConfig.showMessageStats) && isActivelyProcessing && storedReadingStats === null
	);

	$effect(() => {
		if (showLiveReadingStats) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	aria-label="User message with actions"
	class="chat-message-user group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<ChatMessageUserBubble
			content={message.content}
			attachments={message.extra}
			renderMarkdown={true}
		/>

		{#if showStoredReadingStats}
			<!-- Reading stats sourced from the assistant message that followed this turn -->
			<div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
				<div
					class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
				>
					<ChatMessageStatistics
						mode={ChatMessageStatisticsMode.READING}
						promptTokens={storedReadingStats!.promptTokens}
						promptMs={storedReadingStats!.promptMs}
					/>
				</div>
			</div>
		{:else if showLiveReadingStats}
			{@const liveStats = processingState.getLiveProcessingStats()}
			{#if liveStats}
				<div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
					<div
						class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
					>
						<ChatMessageStatistics
							mode={ChatMessageStatisticsMode.READING}
							isLive
							promptTokens={liveStats.tokensProcessed}
							promptMs={liveStats.timeMs}
						/>
					</div>
				</div>
			{/if}
		{/if}

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActionIcons
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onForkConversation}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
