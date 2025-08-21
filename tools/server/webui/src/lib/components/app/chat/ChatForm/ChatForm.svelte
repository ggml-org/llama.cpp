<script lang="ts">
	import { ChatAttachmentsList } from '$lib/components/app';
	import { ChatFormActions, ChatFormFileInputInvisible, ChatFormHelperText, ChatFormTextarea } from '$lib/components/app';
	import { inputClasses } from '$lib/constants/input-classes';
	import { onMount } from 'svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { 
		AudioRecorder, 
		convertToWav, 
		createAudioFile, 
		isAudioRecordingSupported 
	} from '$lib/utils/audio-recording';
	import { 
		TextMimeType, 
		ImageExtension, 
		ImageMimeType, 
		AudioExtension,
		AudioMimeType,
		PdfExtension, 
		PdfMimeType, 
		TextExtension 
	} from '$lib/constants/supported-file-types';

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
		uploadedFiles = $bindable([]),
	}: Props = $props();

	const currentConfig = $derived(config());
	const pasteLongTextToFileLength = $derived(Number(currentConfig.pasteLongTextToFileLen) || 2500);

	let audioRecorder: AudioRecorder | undefined;
	let isRecording = $state(false);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let message = $state('');
	let previousIsLoading = $state(isLoading);
	let recordingSupported = $state(false);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);
	let fileAcceptString = $state<string | undefined>(undefined);

	async function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();

			if ((!message.trim() && uploadedFiles.length === 0) || disabled || isLoading) return;

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

	function handleFileSelect(files: File[]) {
		onFileUpload?.(files);
	}

	function handleFileUpload(fileType?: 'image' | 'audio' | 'pdf' | 'file') {
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

	function getAcceptStringForFileType(fileType: 'image' | 'audio' | 'file' | 'pdf'): string {
		switch (fileType) {
			case 'image':
				return [
					...Object.values(ImageExtension),
					...Object.values(ImageMimeType)
				].join(',');
			case 'audio':
				return [
					...Object.values(AudioExtension),
					...Object.values(AudioMimeType)
				].join(',');
			case 'pdf':
				return [
					...Object.values(PdfExtension),
					...Object.values(PdfMimeType)
				].join(',');
			case 'file':
				return [
					...Object.values(TextExtension),
					TextMimeType.PLAIN
				].join(',');
			default:
				return '';
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

		const text = event.clipboardData.getData(TextMimeType.PLAIN);

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: TextMimeType.PLAIN
			});

			onFileUpload?.([textFile]);
		}
	}

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if ((!message.trim() && uploadedFiles.length === 0) || disabled || isLoading) return;

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

	function handleStop() {
		onStop?.();
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

	onMount(() => {
		textareaRef?.focus();
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			textareaRef?.focus();
		}

		previousIsLoading = isLoading;
	});
</script>

<ChatFormFileInputInvisible bind:this={fileInputRef} bind:accept={fileAcceptString} onFileSelect={handleFileSelect} />

<form
	onsubmit={handleSubmit}
	class="{inputClasses} border-radius-bottom-none mx-auto max-w-[48rem] overflow-hidden rounded-3xl backdrop-blur-md {className}"
>
	<ChatAttachmentsList bind:uploadedFiles {onFileRemove} class="mb-3 px-5 pt-5" />

	<div
		class="flex-column relative min-h-[48px] items-center rounded-3xl px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
		onpaste={handlePaste}
	>
		<ChatFormTextarea
			bind:this={textareaRef}
			bind:value={message}
			onKeydown={handleKeydown}
			disabled={disabled}
		/>

		<ChatFormActions
			canSend={message.trim().length > 0 || uploadedFiles.length > 0}
			{disabled}
			{isLoading}
			{isRecording}
			onFileUpload={handleFileUpload}
			onMicClick={handleMicClick}
			onStop={handleStop}
		/>
	</div>
</form>

<ChatFormHelperText show={showHelperText} />
