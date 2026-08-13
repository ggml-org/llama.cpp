<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageAgenticContent,
		ChatMessageAssistantModel,
		ChatMessageAssistantProcessingInfo,
		ChatMessageAssistantRawOutput,
		ChatMessageAssistantStatistics,
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

	// during a router auto-load the message has no model yet: target the model frozen in the
	// persisted stream state (survives a reload), then fall back to the dropdown selection
	let loadTargetModel = $derived(
		message.model ?? chatStore.getResumeModel(message.convId) ?? modelsStore.selectedModelName
	);
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
		<ChatMessageAssistantProcessingInfo {modelLoadingText} {processingState} position="top" />
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		{#if showRawOutput}
			<ChatMessageAssistantRawOutput {message} {toolMessages} />
		{:else}
			<ChatMessageAgenticContent
				{message}
				{toolMessages}
				isStreaming={chatStore.isStreaming()}
				{isLastAssistantMessage}
			/>
		{/if}
	{/if}

	{#if showProcessingInfoBottom}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} {processingState} position="bottom" />
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
					{message}
					isLoading={chatStore.isLoading}
					{processingState}
					showMessageStats={currentConfig.showMessageStats}
				/>
			</div>
		</div>
	{/if}

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActionIcons
			role={MessageRole.ASSISTANT}
			justify="start"
			actionsPosition="left"
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
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
</style>
