<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageAgenticContent,
		ChatMessageAssistantModel,
		ChatMessageAssistantProcessingInfo,
		ChatMessageAssistantRawOutput,
		ChatMessageAssistantStatistics,
		ChatMessageContextMenu,
		ChatMessageEditForm
	} from '$lib/components/app';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, modelsStore, serverStore, settingsStore } from '$lib/stores';
	import { modelLoadProgressText } from '$lib/utils';
	import { hasAgenticContent } from '$lib/utils';

	interface Props {
		class?: string;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		onContinue?: () => void;
		onRegenerate: (modelOverride?: string) => void;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		isLastAssistantMessage = false,
		message,
		onContinue,
		onRegenerate,
		textareaElement = $bindable(),
		toolMessages = []
	}: Props = $props();

	// Get edit context
	const editCtx = getChatMessageEditContext();

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const processingState = useProcessingState();

	let currentConfig = $derived(settingsStore.config);
	let isRouter = $derived(serverStore.isRouterMode);

	let showRawOutput = $state(false);

	let displayedModel = $derived(message.model ?? null);

	let isCurrentlyLoading = $derived(chatStore.isLoading);
	let isStreaming = $derived(chatStore.isStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);
	// A compaction summary reports its own progress on the compaction row, so
	// this turn stays quiet while it runs instead of duplicating the indicator.
	let isCompacting = $derived(chatStore.isCompacting(message.convId));

	// during a router auto-load the message has no model yet: target the model frozen in the
	// persisted stream state (survives a reload), then fall back to the dropdown selection
	let loadTargetModel = $derived(
		message.model ?? chatStore.getResumeModel(message.convId) ?? modelsStore.selectedModelName
	);
	let modelLoadProgress = $derived(
		isRouter && loadTargetModel ? modelsStore.status.getLoadProgress(loadTargetModel) : null
	);
	let modelLoadingText = $derived(modelLoadProgressText(modelLoadProgress));

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			!isCompacting &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			!isCompacting &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	let assistantEl: HTMLDivElement | undefined = $state();
	let lastUserMessageHeight = $state(0);
	let assistantMarginTop = $state(0);

	// The measured CSS vars feed the :last-child min-height rule only, so only
	// the last assistant message needs them. Reading isLastAssistantMessage
	// here also re-runs the effect when this message stops being the last.
	$effect(() => {
		if (!assistantEl || !isLastAssistantMessage) return;

		assistantMarginTop = Math.round(parseFloat(getComputedStyle(assistantEl).marginTop));

		const chatMessageEl = assistantEl.closest('.chat-message');
		const previousChatMessage = chatMessageEl?.previousElementSibling;
		const userMessageEl = previousChatMessage?.querySelector(
			'.chat-message-user'
		) as HTMLElement | null;

		if (!previousChatMessage || !userMessageEl) {
			lastUserMessageHeight = 0;

			return;
		}

		// a first exchange opened only by synthetic rows (e.g. the working
		// directory notice) must fit the screen as well, so they count as part
		// of the user turn; real earlier messages still scroll away above it
		const leadingRows: Element[] = [];

		for (
			let row = previousChatMessage.previousElementSibling;
			row;
			row = row.previousElementSibling
		) {
			leadingRows.unshift(row);
		}

		const leadingRowsAreSynthetic =
			leadingRows.length > 0 &&
			leadingRows.every((row) => row.classList.contains('chat-message--synthetic'));
		const updateHeight = () => {
			const rect = userMessageEl.getBoundingClientRect();
			const marginTop = Math.round(parseFloat(getComputedStyle(userMessageEl).marginTop));
			const leadingHeight = leadingRowsAreSynthetic
				? previousChatMessage.getBoundingClientRect().top -
					leadingRows[0].getBoundingClientRect().top
				: 0;

			lastUserMessageHeight = Math.round(rect.height + marginTop + leadingHeight);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);

		resizeObserver.observe(userMessageEl);

		if (leadingRowsAreSynthetic) {
			for (const row of leadingRows) resizeObserver.observe(row);
		}

		return () => {
			resizeObserver.disconnect();
		};
	});

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	bind:this={assistantEl}
	style:--assistant-margin-top={assistantMarginTop > 0 ? `${assistantMarginTop}px` : undefined}
	style:--last-user-message-height={lastUserMessageHeight > 0
		? `${lastUserMessageHeight}px`
		: undefined}
	aria-label="Assistant message with actions"
	class="chat-message-assistant text-md group w-full leading-7.5 {className}"
	role="group"
>
	{#if showProcessingInfoTop}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} position="top" {processingState} />
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<ChatMessageContextMenu
			disabled={!message.timestamp}
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
			{onRegenerate}
			role={MessageRole.ASSISTANT}
		>
			{#if showRawOutput}
				<ChatMessageAssistantRawOutput {message} {toolMessages} />
			{:else}
				<ChatMessageAgenticContent
					{isLastAssistantMessage}
					isStreaming={chatStore.isStreaming()}
					{message}
					{toolMessages}
				/>
			{/if}
		</ChatMessageContextMenu>
	{/if}

	{#if showProcessingInfoBottom}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} position="bottom" {processingState} />
	{/if}

	{#if displayedModel}
		<div class="info my-6 grid gap-4 tabular-nums">
			<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
				<ChatMessageAssistantModel
					{displayedModel}
					isLoading={chatStore.isLoading}
					{isRouter}
					{onRegenerate}
				/>

				<ChatMessageAssistantStatistics
					isLoading={chatStore.isLoading}
					{message}
					{processingState}
					showMessageStats={currentConfig.showMessageStats}
				/>
			</div>
		</div>
	{/if}

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActionIcons
			actionsPosition="left"
			justify="start"
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
			onRawOutputToggle={(enabled) => (showRawOutput = enabled)}
			{onRegenerate}
			rawOutputEnabled={showRawOutput}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
		/>
	{/if}
</div>

<style>
	:global(.chat-message):last-child .chat-message-assistant {
		--assistant-min-height-offset: calc(
			var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
				var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
				var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
		);
		min-height: calc(100dvh - var(--assistant-min-height-offset));

		@media (width > 768px) {
			--assistant-min-height-offset: calc(
				var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
					var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
					var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
			);
		}
	}
</style>
