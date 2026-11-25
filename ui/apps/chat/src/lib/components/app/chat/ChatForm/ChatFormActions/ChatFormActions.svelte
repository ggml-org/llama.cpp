<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { LoaderCircle, SkipForward, Square } from '@lucide/svelte';
	import {
		ChatFormActionRecord,
		ChatFormActionsAdd,
		ChatFormActionSubmit,
		ChatFormContextGauge,
		ModelsSelector
	} from '#lib/components/app/index.ts';
	import {
		ICON_CLASS_DEFAULT,
		SET_WORKING_DIRECTORY_LABEL,
		VOICE_MODE_MESSAGE
	} from '#lib/constants/index.ts';
	import { setChatFormActionsContext } from '#lib/contexts/index.ts';
	import { useChatFormModel } from '#lib/hooks/use-chat-form-model.svelte.ts';
	import { useShowContextGauge } from '#lib/hooks/use-context-gauge.svelte.ts';
	import { ChatService } from '#lib/services/index.ts';
	import { conversationsStore, deviceStore, settingsStore, toolsStore } from '#lib/stores/index.ts';
	import { abbreviateWorkingDir } from '#lib/utils/index.ts';

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		/** Lay the actions out in the row around them, the add button first, the rest last. */
		inline?: boolean;
		isLoading?: boolean;
		isReasoning?: boolean;
		/** A send was asked for and waits for the dictated text to be refined. */
		isSendWaiting?: boolean;
		/** The dictated text in the message is still being refined. */
		isVoiceRefining?: boolean;
		/** Why the microphone can't record, shown on its button. */
		micWarning?: string | null;
		/** Voice input: whether the microphone shows, what it does, and why it can't. */
		voice?: {
			action: string;
			hint: string | null;
			show: boolean;
			unavailableReason: string | null;
			/** The pointer or the focus came to the microphone button, a press may follow */
			prepare?: () => void;
			/** It left without a press */
			unprepare?: () => void;
		};
		showAddButton?: boolean;
		showModelSelector?: boolean;
		/** The chat's working directory, which the + menu names. */
		workingDirectory?: string | null;
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
		onWorkingDirectoryClick?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		inline = false,
		isLoading = false,
		isReasoning = false,
		isSendWaiting = false,
		isVoiceRefining = false,
		micWarning = null,
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick,
		onWorkingDirectoryClick,
		showAddButton = true,
		showModelSelector = true,
		voice,
		workingDirectory = null
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

	// the microphone sits beside the send or stop button: on an empty message when the setting
	// asks for it (disabled, saying why, when no model can take a recording), and while a reply
	// streams or the message has text or a recording, so more can be said without clearing it
	let showStop = $derived(isLoading && !canSend);
	let showRecord = $derived.by(() => {
		if (!voice?.show) return false;

		if (showStop || canSend) return !voice.unavailableReason;

		return currentConfig.autoMicOnEmpty;
	});

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
		},
		get onWorkingDirectoryClick() {
			return onWorkingDirectoryClick;
		},
		get workingDirectoryLabel() {
			return workingDirectory
				? abbreviateWorkingDir(workingDirectory, toolsStore.serverHome)
				: SET_WORKING_DIRECTORY_LABEL;
		}
	});
</script>

<div
	style:container-type={inline ? undefined : 'inline-size'}
	class={[
		inline ? 'contents' : 'flex w-full items-center gap-3',
		className,
		!showAddButton && 'justify-end'
	]}
>
	{#if showAddButton}
		<div class={['flex items-center gap-2', inline ? 'order-first' : 'mr-auto']}>
			<ChatFormActionsAdd />
		</div>
	{/if}

	<div class="flex items-center gap-1.5">
		<!-- a phone shows the gauge pinned to the chat screen's top right instead -->
		{#if showContextGauge.current && !deviceStore.isMobile}
			<ChatFormContextGauge />
		{/if}

		{#if showModelSelector}
			<!-- the reasoning effort sits in the model picker, under the models -->
			<ModelsSelector
				bind:this={selectorModelRef}
				{disabled}
				forceForegroundText
				showReasoning
				useGlobalSelection
			/>
		{/if}
	</div>

	{#if isVoiceRefining}
		<span
			class="flex items-center gap-1.5 text-caption whitespace-nowrap text-muted-foreground"
			data-slot="voice-refining"
			role="status"
		>
			<LoaderCircle class="size-icon-sm animate-spin" />

			{isSendWaiting ? VOICE_MODE_MESSAGE.sendAfterRefining : VOICE_MODE_MESSAGE.refining}
		</span>
	{/if}

	{#if isReasoning && !canSend}
		<Button
			class="group size-8 rounded-full p-0 max-md:size-9"
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

	{#snippet record(filled: boolean)}
		<ChatFormActionRecord
			action={voice?.action}
			{disabled}
			{filled}
			hint={voice?.hint}
			{micWarning}
			{onMicClick}
			onPrepare={voice?.prepare}
			onPrepareEnd={voice?.unprepare}
			secondary
			unavailableReason={voice?.unavailableReason}
		/>
	{/snippet}

	{#if showRecord && !showStop}
		{@render record(false)}
	{/if}

	{#if showStop}
		<Button
			class="group size-8 rounded-full p-0 max-md:size-9 hover:bg-destructive/10!"
			onclick={onStop}
			type="button"
			variant="secondary"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="size-icon-md fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && (showModelSelector ? hasModelSelected && isSelectedModelInCache : true)}
			{disabled}
			loading={isSendWaiting}
			showErrorState={showModelSelector && hasModelSelected && !isSelectedModelInCache}
			tooltipLabel={submitTooltip}
		/>
	{/if}

	<!-- while a reply streams, the microphone is the last button, filled like stop -->
	{#if showRecord && showStop}
		{@render record(true)}
	{/if}
</div>
