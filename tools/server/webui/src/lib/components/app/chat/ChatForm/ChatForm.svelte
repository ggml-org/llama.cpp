<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatFormHelperText, ChatFormInputArea } from '$lib/components/app';
	import { onMount } from 'svelte';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSystemPromptAdd?: () => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		placeholder = 'Type a message...',
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onValueChange
	}: Props = $props();

	let inputAreaRef: ChatFormInputArea | undefined = $state(undefined);
	let message = $state('');
	let previousIsLoading = $state(isLoading);

	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));

	async function handleSubmit() {
		if (
			(!message.trim() && uploadedFiles.length === 0) ||
			disabled ||
			isLoading ||
			hasLoadingAttachments
		)
			return;

		if (!inputAreaRef?.checkModelSelected()) return;

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];

		inputAreaRef?.resetHeight();

		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
		}
	}

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	onMount(() => {
		setTimeout(() => inputAreaRef?.focus(), 10);
	});

	afterNavigate(() => {
		setTimeout(() => inputAreaRef?.focus(), 10);
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(() => inputAreaRef?.focus(), 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div class="relative mx-auto max-w-[48rem]">
	<ChatFormInputArea
		bind:this={inputAreaRef}
		bind:value={message}
		bind:uploadedFiles
		class={className}
		{disabled}
		{isLoading}
		showMcpPromptButton={true}
		onFilesAdd={handleFilesAdd}
		{onStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={onSystemPromptAdd}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>

<ChatFormHelperText show={showHelperText} />
