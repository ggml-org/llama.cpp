<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageStatistics,
		ChatMessageUserBubble
	} from '$lib/components/app/chat';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { ChatMessageStatisticsMode, MessageRole } from '$lib/enums';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, settingsStore } from '$lib/stores';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		isLastUserMessage?: boolean;
		nextAssistantMessage?: DatabaseMessage | null;
	}

	let {
		class: className = '',
		isLastUserMessage = false,
		message,
		nextAssistantMessage = null
	}: Props = $props();

	// Get contexts
	const editCtx = getChatMessageEditContext();
	const processingState = useProcessingState();

	const currentConfig = $derived(settingsStore.config);
	const isActivelyProcessing = $derived(isLastUserMessage && chatStore.isLoading);

	// For agentic turns, prefer the cumulative agentic.llm totals over per-call timings.
	let storedReadingStats = $derived.by(() => {
		const timings = nextAssistantMessage?.timings;

		if (!timings?.prompt_n || !timings?.prompt_ms) return null;

		const agentic = timings.agentic;

		return {
			promptMs: agentic ? agentic.llm.prompt_ms : timings.prompt_ms,
			promptTokens: agentic ? agentic.llm.prompt_n : timings.prompt_n
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
			attachments={message.extra}
			content={message.content}
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
						promptMs={storedReadingStats!.promptMs}
						promptTokens={storedReadingStats!.promptTokens}
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
							isLive
							mode={ChatMessageStatisticsMode.READING}
							promptMs={liveStats.timeMs}
							promptTokens={liveStats.tokensProcessed}
						/>
					</div>
				</div>
			{/if}
		{/if}

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActionIcons actionsPosition="right" justify="end" role={MessageRole.USER} />
			</div>
		{/if}
	{/if}
</div>
