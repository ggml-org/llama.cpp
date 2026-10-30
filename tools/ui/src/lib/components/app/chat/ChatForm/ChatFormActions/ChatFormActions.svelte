<script lang="ts">
	import { SkipForward, Square } from '@lucide/svelte';
	import {
		ChatFormActionReasoning,
		ChatFormActionRecord,
		ChatFormActionsAdd,
		ChatFormActionSubmit,
		ChatFormContextGauge,
		ModelsSelector
	} from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { setChatFormActionsContext } from '$lib/contexts';
	import { FileTypeCategory } from '$lib/enums';
	import { useChatFormModel } from '$lib/hooks/use-chat-form-model.svelte';
	import { useShowContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import { ChatService } from '$lib/services';
	import { conversationsStore, deviceStore, settingsStore } from '$lib/stores';
	import { getFileTypeCategory } from '$lib/utils';

	interface Props {
		canSend?: boolean;
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
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isReasoning = false,
		isRecording = false,
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = []
	}: Props = $props();

	let currentConfig = $derived(settingsStore.config);

	const formModel = useChatFormModel();

	let hasAudioModality = $derived(formModel.hasAudioModality);
	let hasVideoModality = $derived(formModel.hasVideoModality);
	let hasVisionModality = $derived(formModel.hasVisionModality);
	let hasModelSelected = $derived(formModel.hasModelSelected);
	let isSelectedModelInCache = $derived(formModel.isSelectedModelInCache);
	let submitTooltip = $derived.by(() => {
		if (!hasModelSelected) return 'Please select a model first';

		if (!isSelectedModelInCache) return 'Selected model is not available, please select another';

		return '';
	});

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !canSend && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let selectorModelRef: ModelsSelector | undefined = $state(undefined);

	export function openModelSelector() {
		selectorModelRef?.open();
	}
	// the streaming assistant message carries both the completion id and the model that
	// produced it, targeting reasoning control from the same source keeps them consistent
	let activeMessage = $derived(
		conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1]
	);

	const showContextGauge = useShowContextGauge();

	setChatFormActionsContext({
		get disabled() {
			return disabled;
		},
		get hasAudioModality() {
			return hasAudioModality;
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
		<!-- a phone shows the gauge pinned to the chat screen's top right instead -->
		{#if showContextGauge.current && !deviceStore.isMobile}
			<ChatFormContextGauge />
		{/if}

		{#if showModelSelector}
			<ChatFormActionReasoning />

			<ModelsSelector
				bind:this={selectorModelRef}
				{disabled}
				forceForegroundText
				useGlobalSelection
			/>
		{/if}
	</div>

	{#if isReasoning && !canSend}
		<Button
			class="group h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9"
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

	{#if isLoading && !canSend}
		<Button
			class="group h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9 hover:bg-destructive/10!"
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
