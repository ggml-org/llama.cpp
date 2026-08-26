<script lang="ts">
	import { SkipForward, Square } from '@lucide/svelte';
	import { page } from '$app/state';
	import {
		ChatFormActionModels,
		ChatFormActionRecord,
		ChatFormActionsAdd,
		ChatFormActionSubmit,
		ChatFormContextGauge
	} from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { setChatFormActionsContext } from '$lib/contexts';
	import { FileTypeCategory, MessageRole } from '$lib/enums';
	import { ChatService } from '$lib/services';
	import { chatStore, conversationsStore, mcpStore, settingsStore } from '$lib/stores';
	import { getFileTypeCategory } from '$lib/utils';

	interface Props {
		canSend?: boolean;
		canSubmit?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isReasoning?: boolean;
		isRecording?: boolean;
		showAddButton?: boolean;
		showModelSelector?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
		onMcpSettingsClick?: () => void;
	}

	let {
		canSend = false,
		canSubmit = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isReasoning = false,
		isRecording = false,
		onFileUpload,
		onMcpPromptClick,
		onMcpResourcesClick,
		onMcpSettingsClick,
		onMicClick,
		onStop,
		onSystemPromptClick,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = []
	}: Props = $props();

	let currentConfig = $derived(settingsStore.config);

	let hasMcpPromptsSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.preferences.getAllMcpServerOverrides();

		return mcpStore.hasPromptsCapability(perChatOverrides);
	});

	let hasMcpResourcesSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.preferences.getAllMcpServerOverrides();

		return mcpStore.hasResourcesCapability(perChatOverrides);
	});

	let hasAudioModality = $state(false);
	let hasVideoModality = $state(false);
	let hasVisionModality = $state(false);
	let hasModelSelected = $state(false);
	let isSelectedModelInCache = $state(true);
	let submitTooltip = $state('');

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !canSubmit && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let selectorModelRef: ChatFormActionModels | undefined = $state(undefined);

	export function openModelSelector() {
		selectorModelRef?.open();
	}
	// the streaming assistant message carries both the completion id and the model that
	// produced it, targeting reasoning control from the same source keeps them consistent
	let activeMessage = $derived(
		conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1]
	);

	let hasProcessedTokens = $derived.by(() => {
		if (!page.params.id) return false;

		const messages = conversationsStore.activeMessages as DatabaseMessage[];

		let totalHistoricalTokens = 0;

		for (const m of messages) {
			if (m.role !== MessageRole.ASSISTANT) continue;

			const timings = m.timings;

			if (!timings) continue;

			const agenticLlm = timings.agentic?.llm;

			if (agenticLlm?.prompt_n != null || agenticLlm?.predicted_n != null) {
				totalHistoricalTokens += (agenticLlm?.prompt_n ?? 0) + (agenticLlm?.predicted_n ?? 0);
			} else {
				totalHistoricalTokens += (timings.prompt_n ?? 0) + (timings.predicted_n ?? 0);
			}
		}

		if (totalHistoricalTokens > 0) return true;

		if (!chatStore.isLoading && !chatStore.isStreaming()) return false;

		const processingState = chatStore.processing.activeState;

		if (!processingState) return false;

		const livePromptTokens = Math.max(
			processingState.promptTokens ?? 0,
			processingState.promptProgress?.processed ?? 0
		);
		const liveOutputTokens = processingState.outputTokensUsed ?? 0;

		return livePromptTokens > 0 || liveOutputTokens > 0;
	});

	setChatFormActionsContext({
		get disabled() {
			return disabled;
		},
		get hasAudioModality() {
			return hasAudioModality;
		},
		get hasMcpPromptsSupport() {
			return hasMcpPromptsSupport;
		},
		get hasMcpResourcesSupport() {
			return hasMcpResourcesSupport;
		},
		get hasVideoModality() {
			return hasVideoModality;
		},
		get hasVisionModality() {
			return hasVisionModality;
		},
		get onFileUpload() {
			return onFileUpload;
		},
		get onMcpPromptClick() {
			return onMcpPromptClick;
		},
		get onMcpResourcesClick() {
			return onMcpResourcesClick;
		},
		get onMcpSettingsClick() {
			return onMcpSettingsClick;
		},
		get onSystemPromptClick() {
			return onSystemPromptClick;
		}
	});
</script>

<div
	class="flex w-full items-center gap-3 {className} {showAddButton ? '' : 'justify-end'}"
	style="container-type: inline-size"
>
	{#if showAddButton}
		<div class="mr-auto flex items-center gap-2">
			<ChatFormActionsAdd />
		</div>
	{/if}

	<div class="flex items-center gap-1.5">
		{#if hasProcessedTokens}
			<ChatFormContextGauge />
		{/if}

		{#if showModelSelector}
			<ChatFormActionModels
				bind:hasAudioModality
				bind:hasModelSelected
				bind:hasVideoModality
				bind:hasVisionModality
				bind:isSelectedModelInCache
				bind:submitTooltip
				bind:this={selectorModelRef}
				{disabled}
				forceForegroundText
				useGlobalSelection
			/>
		{/if}
	</div>

	{#if isReasoning}
		<Button
			class="group h-8 w-8 rounded-full p-0"
			onclick={() =>
				ChatService.stopReasoning(activeMessage?.completionId ?? '', activeMessage?.model)}
			title="Skip reasoning"
			type="button"
			variant="secondary"
		>
			<span class="sr-only">Skip reasoning</span>

			<SkipForward
				class="{ICON_CLASS_DEFAULT} stroke-muted-foreground group-hover:stroke-foreground"
			/>
		</Button>
	{/if}

	{#if isLoading && !canSubmit}
		<Button
			class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
			onclick={onStop}
			type="button"
			variant="secondary"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && (showModelSelector ? hasModelSelected && isSelectedModelInCache : true)}
			{disabled}
			showErrorState={showModelSelector && hasModelSelected && !isSelectedModelInCache}
			tooltipLabel={submitTooltip}
		/>
	{/if}
</div>
