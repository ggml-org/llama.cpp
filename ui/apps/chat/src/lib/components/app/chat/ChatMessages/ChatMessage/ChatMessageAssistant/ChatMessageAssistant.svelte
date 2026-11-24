<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageAgentTrace,
		ChatMessageAssistantModel,
		ChatMessageAssistantProcessingInfo,
		ChatMessageAssistantRawOutput,
		ChatMessageAssistantResponse,
		ChatMessageAssistantStatistics,
		ChatMessageContextMenu,
		ChatMessageEditForm
	} from '#lib/components/app/index.ts';
	import { SETTINGS_KEYS } from '#lib/constants/index.ts';
	import { getChatMessageEditContext } from '#lib/contexts/index.ts';
	import { MessageRole } from '#lib/enums/index.ts';
	import { useProcessingState } from '#lib/hooks/use-processing-state.svelte.ts';
	import { chatStore, modelsStore, serverStore, settingsStore } from '#lib/stores/index.ts';
	import { hasAgenticContent, modelLoadProgressText } from '#lib/utils/index.ts';

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

	const processingState = useProcessingState();

	let currentConfig = $derived(settingsStore.config);
	let isRouter = $derived(serverStore.isRouterMode);

	let showRawOutput = $state(false);

	// A reply that ran the agentic loop ( it called tools ) is an Agent Trace;
	// a basic chat reply is just its reasoning and its answer. A live reply
	// becomes a trace the moment its first tool call streams in.
	const isAgentTrace = $derived(hasAgenticContent(message, toolMessages));

	let isCurrentlyLoading = $derived(chatStore.isLoading);
	let isStreaming = $derived(chatStore.isStreaming());
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

	// The model badge shows from the start of a live reply, before the first
	// chunk names the model, so the stats row does not pop in under the status
	let displayedModel = $derived(
		message.model ?? (isLastAssistantMessage && isActivelyProcessing ? loadTargetModel : null)
	);

	// What the server is doing while nothing is generated yet; null when this
	// message is not the live one. Empty while tokens are streaming.
	let statusText = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			!isCompacting &&
			isLastAssistantMessage
			? modelLoadingText ||
					processingState.getPromptProgressText() ||
					processingState.getProcessingMessage()
			: null
	);

	// the latest exchange keeps its stats and actions in view; earlier messages
	// show them on hover, and the setting can keep their stats in view as well
	let hidesStatsUntilHover = $derived(
		!isLastAssistantMessage && !currentConfig[SETTINGS_KEYS.ALWAYS_SHOW_MESSAGE_STATS]
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
		if (statusText !== null) {
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

				{#if statusText}
					<ChatMessageAssistantProcessingInfo text={statusText} />
				{/if}
			{:else if isAgentTrace}
				<ChatMessageAgentTrace
					{isLastAssistantMessage}
					isStreaming={chatStore.isStreaming()}
					{message}
					{statusText}
					{toolMessages}
				/>
			{:else}
				<ChatMessageAssistantResponse
					{isLastAssistantMessage}
					isStreaming={chatStore.isStreaming()}
					{message}
					{statusText}
				/>
			{/if}
		</ChatMessageContextMenu>
	{/if}

	{#if displayedModel}
		<div class={['info my-4 grid gap-4 tabular-nums', hidesStatsUntilHover && 'message-meta']}>
			<div class="inline-flex flex-wrap items-start gap-2 text-caption text-muted-foreground">
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
		<div class={isLastAssistantMessage ? undefined : 'message-meta'}>
			<ChatMessageActionIcons
				actionsPosition="left"
				justify="start"
				onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
				onRawOutputToggle={(enabled) => (showRawOutput = enabled)}
				{onRegenerate}
				rawOutputEnabled={showRawOutput}
				showRawOutputSwitch={currentConfig.showRawOutputSwitch}
			/>
		</div>
	{/if}
</div>

<style>
	/* The reply's minimum height decides where a sent message rests once it
	   glides up, and the top offset adds room above it. A phone's floating
	   header covers the top of the page, so there it rests below the header. */
	:global(.chat-message):last-child .chat-message-assistant {
		--sent-message-top-offset: var(--page-top);
		--assistant-min-height-offset: calc(
			var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
				var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
				var(--assistant-margin-top, 1rem) + var(--chat-tabs-offset, 0px) +
				var(--sent-message-top-offset)
		);
		min-height: calc(100dvh - var(--assistant-min-height-offset));

		@media (width > 768px) {
			--sent-message-top-offset: 0.5rem;
			--assistant-min-height-offset: calc(
				var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
					var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
					var(--assistant-margin-top, 1rem) + var(--chat-tabs-offset, 0px) +
					var(--sent-message-top-offset)
			);
		}
	}
</style>
