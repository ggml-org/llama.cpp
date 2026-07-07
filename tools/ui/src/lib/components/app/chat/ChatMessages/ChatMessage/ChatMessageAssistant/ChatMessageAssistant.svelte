<script lang="ts">
	import {
		ChatMessageAgenticContent,
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageStatistics,
		ModelBadge,
		ModelsSelectorDropdown
	} from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { copyToClipboard, deriveAgenticSections, modelLoadProgressText } from '$lib/utils';
	import { AgenticSectionType } from '$lib/enums';
	import { REASONING_TAGS } from '$lib/constants/agentic';
	import { tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { MessageRole, ChatMessageStatsView } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	import { hasAgenticContent } from '$lib/utils';

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		messageContent: string | undefined;
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		showDeleteDialog: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		deletionInfo,
		isLastAssistantMessage = false,
		message,
		toolMessages = [],
		messageContent,
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onForkConversation,
		onNavigateToSibling,
		onRegenerate,
		onShowDeleteDialogChange,
		showDeleteDialog,
		siblingInfo = null,
		textareaElement = $bindable()
	}: Props = $props();

	// Get edit context
	const editCtx = getMessageEditContext();

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const processingState = useProcessingState();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	let showRawOutput = $state(false);

	let rawOutputContent = $derived.by(() => {
		const sections = deriveAgenticSections(message, toolMessages, [], false);
		const parts: string[] = [];

		for (const section of sections) {
			switch (section.type) {
				case AgenticSectionType.REASONING:
				case AgenticSectionType.REASONING_PENDING:
					parts.push(`${REASONING_TAGS.START}\n${section.content}\n${REASONING_TAGS.END}`);
					break;

				case AgenticSectionType.TEXT:
					parts.push(section.content);
					break;

				case AgenticSectionType.TOOL_CALL:
				case AgenticSectionType.TOOL_CALL_PENDING:
				case AgenticSectionType.TOOL_CALL_STREAMING: {
					const callObj: Record<string, unknown> = { name: section.toolName };

					if (section.toolArgs) {
						try {
							callObj.arguments = JSON.parse(section.toolArgs);
						} catch {
							callObj.arguments = section.toolArgs;
						}
					}

					parts.push(JSON.stringify(callObj, null, 2));

					if (section.toolResult) {
						parts.push(`[Tool Result]\n${section.toolResult}`);
					}

					break;
				}
			}
		}

		return parts.join('\n\n\n');
	});

	let activeStatsView = $state<ChatMessageStatsView>(ChatMessageStatsView.GENERATION);
	let statsContainerEl: HTMLDivElement | undefined = $state();

	function getScrollParent(el: HTMLElement): HTMLElement | null {
		let parent = el.parentElement;
		while (parent) {
			const style = getComputedStyle(parent);
			if (/(auto|scroll)/.test(style.overflowY)) {
				return parent;
			}
			parent = parent.parentElement;
		}
		return null;
	}

	async function handleStatsViewChange(view: ChatMessageStatsView) {
		const el = statsContainerEl;
		if (!el) {
			activeStatsView = view;

			return;
		}

		const scrollParent = getScrollParent(el);
		if (!scrollParent) {
			activeStatsView = view;

			return;
		}

		const yBefore = el.getBoundingClientRect().top;

		activeStatsView = view;

		await tick();

		const delta = el.getBoundingClientRect().top - yBefore;
		if (delta !== 0) {
			scrollParent.scrollTop += delta;
		}

		// Correct any drift after browser paint
		requestAnimationFrame(() => {
			const drift = el.getBoundingClientRect().top - yBefore;

			if (Math.abs(drift) > 1) {
				scrollParent.scrollTop += drift;
			}
		});
	}

	let highlightAgenticTurns = $derived(
		isAgentic &&
			(currentConfig.alwaysShowAgenticTurns || activeStatsView === ChatMessageStatsView.SUMMARY)
	);

	let displayedModel = $derived(message.model ?? null);

	// model being switched to while it loads, so the selector bar tracks it
	let pendingModel = $state<string | null>(null);

	let isCurrentlyLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);

	// during a router auto-load the message has no model yet, so target the selected one
	let loadTargetModel = $derived(message.model ?? modelsStore.selectedModelName);
	let modelLoadProgress = $derived(
		isRouter && loadTargetModel ? modelsStore.getLoadProgress(loadTargetModel) : null
	);
	let modelLoadingText = $derived(modelLoadProgressText(modelLoadProgress));

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	let assistantEl: HTMLDivElement | undefined = $state();
	let lastUserMessageHeight = $state(0);
	let assistantMarginTop = $state(0);

	$effect(() => {
		if (!assistantEl) return;

		assistantMarginTop = Math.round(parseFloat(getComputedStyle(assistantEl).marginTop));

		const chatMessageEl = assistantEl.closest('.chat-message');
		const previousChatMessage = chatMessageEl?.previousElementSibling;
		const userMessageEl = previousChatMessage?.querySelector(
			'.chat-message-user'
		) as HTMLElement | null;

		if (!userMessageEl) {
			lastUserMessageHeight = 0;
			return;
		}

		const updateHeight = () => {
			const rect = userMessageEl.getBoundingClientRect();
			const marginTop = Math.round(parseFloat(getComputedStyle(userMessageEl).marginTop));
			lastUserMessageHeight = Math.round(rect.height + marginTop);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);
		resizeObserver.observe(userMessageEl);

		return () => {
			resizeObserver.disconnect();
		};
	});

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	bind:this={assistantEl}
	class="chat-message-assistant text-md group w-full leading-7.5 {className}"
	style:--last-user-message-height={lastUserMessageHeight > 0
		? `${lastUserMessageHeight}px`
		: undefined}
	style:--assistant-margin-top={assistantMarginTop > 0 ? `${assistantMarginTop}px` : undefined}
	role="group"
	aria-label="Assistant message with actions"
>
	{#if showProcessingInfoTop}
		<div class="mt-6 w-full max-w-3xl" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{modelLoadingText ??
						processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else if message.role === MessageRole.ASSISTANT}
		{#if showRawOutput}
			<pre class="raw-output">{rawOutputContent || ''}</pre>
		{:else}
			<ChatMessageAgenticContent
				{message}
				{toolMessages}
				isStreaming={isChatStreaming()}
				{isLastAssistantMessage}
				highlightTurns={highlightAgenticTurns}
			/>
		{/if}
	{:else}
		<div class="text-sm whitespace-pre-wrap">
			{messageContent}
		</div>
	{/if}

	{#if showProcessingInfoBottom}
		<div class="mt-4 w-full max-w-3xl" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{modelLoadingText ??
						processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	<div class="info my-6 grid gap-4 tabular-nums">
		{#if displayedModel}
			<div
				bind:this={statsContainerEl}
				class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground"
			>
				{#if isRouter}
					<ModelsSelectorDropdown
						currentModel={pendingModel ?? displayedModel}
						disabled={isLoading()}
						onModelChange={async (modelId: string, modelName: string) => {
							const status = modelsStore.getModelStatus(modelId);

							if (status !== ServerModelStatus.LOADED) {
								pendingModel = modelId;

								try {
									await modelsStore.loadModel(modelId);
								} finally {
									pendingModel = null;
								}
							}

							onRegenerate(modelName);
							return true;
						}}
					/>
				{:else}
					<ModelBadge model={displayedModel || undefined} onclick={handleCopyModel} />
				{/if}

				{#if currentConfig.showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
					{@const agentic = message.timings.agentic}
					<ChatMessageStatistics
						promptTokens={agentic ? agentic.llm.prompt_n : message.timings.prompt_n}
						promptMs={agentic ? agentic.llm.prompt_ms : message.timings.prompt_ms}
						predictedTokens={agentic ? agentic.llm.predicted_n : message.timings.predicted_n}
						predictedMs={agentic ? agentic.llm.predicted_ms : message.timings.predicted_ms}
						agenticTimings={agentic}
						onActiveViewChange={handleStatsViewChange}
					/>
				{:else if isLoading() && currentConfig.showMessageStats}
					{@const liveStats = processingState.getLiveProcessingStats()}
					{@const genStats = processingState.getLiveGenerationStats()}
					{@const promptProgress = processingState.processingState?.promptProgress}
					{@const isStillProcessingPrompt =
						promptProgress && promptProgress.processed < promptProgress.total}

					{#if liveStats || genStats}
						<ChatMessageStatistics
							isLive
							isProcessingPrompt={!!isStillProcessingPrompt}
							promptTokens={liveStats?.tokensProcessed}
							promptMs={liveStats?.timeMs}
							predictedTokens={genStats?.tokensGenerated}
							predictedMs={genStats?.timeMs}
						/>
					{/if}
				{/if}
			</div>
		{/if}
	</div>

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActionIcons
			role={MessageRole.ASSISTANT}
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
			{onForkConversation}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
			rawOutputEnabled={showRawOutput}
			onRawOutputToggle={(enabled) => (showRawOutput = enabled)}
		/>
	{/if}
</div>

<style>
	:global(.chat-message):last-child .chat-message-assistant {
		--assistant-min-height-offset: calc(
			var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
				var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
				var(--assistant-margin-top, 3rem)
		);
		min-height: calc(100dvh - var(--assistant-min-height-offset));

		@media (width > 768px) {
			--assistant-min-height-offset: calc(
				var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
					var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
					var(--assistant-margin-top, 3rem)
			);
		}
	}

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
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
