<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormFileInputInvisible,
		ChatFormHelperText,
		ChatFormTextarea
	} from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import { config } from '$lib/stores/settings.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { activeMessages } from '$lib/stores/conversations.svelte';
	import {
		FileTypeCategory,
		MimeTypeApplication,
		FileExtensionAudio,
		FileExtensionImage,
		FileExtensionPdf,
		FileExtensionText,
		MimeTypeAudio,
		MimeTypeImage,
		MimeTypeText
	} from '$lib/enums';
	import { isIMEComposing } from '$lib/utils';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		showHelperText = true,
		uploadedFiles = $bindable([])
	}: Props = $props();

	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let currentConfig = $derived(config());
	let fileAcceptString = $state<string | undefined>(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let isRecording = $state(false);
	let message = $state('');
	let pasteLongTextToFileLength = $derived(Number(currentConfig.pasteLongTextToFileLen) || 2500);
	let previousIsLoading = $state(isLoading);
	let recordingSupported = $state(false);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);

	// Check if model is selected (in ROUTER mode)
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);
	let isRouter = $derived(isRouterMode());
	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());

	// Get active model ID for capability detection
	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		// First try user-selected model
		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		// Fallback to conversation model
		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	// State for model props reactivity
	let modelPropsVersion = $state(0);

	// Fetch model props when active model changes (works for both MODEL and ROUTER mode)
	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);
			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	// Derive modalities from active model (works for both MODEL and ROUTER mode)
	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion; // Trigger reactivity on props fetch
			return modelsStore.modelSupportsAudio(activeModelId);
		}

		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion; // Trigger reactivity on props fetch
			return modelsStore.modelSupportsVision(activeModelId);
		}

		return false;
	});

	function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			// Open the model selector
			chatFormActionsRef?.openModelSelector();
			return false;
		}

		return true;
	}

	function getAcceptStringForFileType(fileType: FileTypeCategory): string {
		switch (fileType) {
			case FileTypeCategory.IMAGE:
				return [...Object.values(FileExtensionImage), ...Object.values(MimeTypeImage)].join(',');

			case FileTypeCategory.AUDIO:
				return [...Object.values(FileExtensionAudio), ...Object.values(MimeTypeAudio)].join(',');

			case FileTypeCategory.PDF:
				return [...Object.values(FileExtensionPdf), ...Object.values(MimeTypeApplication)].join(
					','
				);

			case FileTypeCategory.TEXT:
				return [...Object.values(FileExtensionText), MimeTypeText.PLAIN].join(',');

			default:
				return '';
		}
	}

	function handleFileSelect(files: File[]) {
		onFileUpload?.(files);
	}

	function handleFileUpload(fileType?: FileTypeCategory) {
		if (fileType) {
			fileAcceptString = getAcceptStringForFileType(fileType);
		} else {
			fileAcceptString = undefined;
		}

		// Use setTimeout to ensure the accept attribute is applied before opening dialog
		setTimeout(() => {
			fileInputRef?.click();
		}, 10);
	}

	async function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();

			if ((!message.trim() && uploadedFiles.length === 0) || disabled || isLoading) return;

			// Check if model is selected first
			if (!checkModelSelected()) return;

			const messageToSend = message.trim();
			const filesToSend = [...uploadedFiles];

			message = '';
			uploadedFiles = [];

			textareaRef?.resetHeight();

			const success = await onSend?.(messageToSend, filesToSend);

			if (!success) {
				message = messageToSend;
				uploadedFiles = filesToSend;
			}
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFileUpload?.(files);

			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFileUpload?.([textFile]);
		}
	}

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');

			return;
		}

		if (isRecording) {
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFileUpload?.([audioFile]);
				isRecording = false;
			} catch (error) {
				console.error('Failed to stop recording:', error);
				isRecording = false;
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}

	function handleStop() {
		onStop?.();
	}

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if ((!message.trim() && uploadedFiles.length === 0) || disabled || isLoading) return;

		// Check if model is selected first
		if (!checkModelSelected()) return;

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];

		textareaRef?.resetHeight();

		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
		}
	}

	onMount(() => {
		setTimeout(() => textareaRef?.focus(), 10);
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	afterNavigate(() => {
		setTimeout(() => textareaRef?.focus(), 10);
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(() => textareaRef?.focus(), 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<ChatFormFileInputInvisible
	bind:this={fileInputRef}
	bind:accept={fileAcceptString}
	{hasAudioModality}
	{hasVisionModality}
	onFileSelect={handleFileSelect}
/>

<form
	onsubmit={handleSubmit}
	class="{INPUT_CLASSES} border-radius-bottom-none mx-auto max-w-[48rem] overflow-hidden rounded-3xl backdrop-blur-md {disabled
		? 'cursor-not-allowed opacity-60'
		: ''} {className}"
>
	<ChatAttachmentsList
		bind:uploadedFiles
		{onFileRemove}
		limitToSingleRow
		class="py-5"
		style="scroll-padding: 1rem;"
		activeModelId={activeModelId ?? undefined}
	/>

	<div
		class="flex-column relative min-h-[48px] items-center rounded-3xl px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
		onpaste={handlePaste}
	>
		<ChatFormTextarea
			bind:this={textareaRef}
			bind:value={message}
			onKeydown={handleKeydown}
			{disabled}
		/>

		<ChatFormActions
			bind:this={chatFormActionsRef}
			canSend={message.trim().length > 0 || uploadedFiles.length > 0}
			hasText={message.trim().length > 0}
			{disabled}
			{isLoading}
			{isRecording}
			{uploadedFiles}
			onFileUpload={handleFileUpload}
			onMicClick={handleMicClick}
			onStop={handleStop}
		/>
	</div>
</form>

<ChatFormHelperText show={showHelperText} />
