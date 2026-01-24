<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatFormHelperText, ChatFormInputArea } from '$lib/components/app';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		initialMessage?: string;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		onSystemPromptAdd?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		initialMessage = '',
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		onSystemPromptAdd,
		showHelperText = true,
		uploadedFiles = $bindable([])
	}: Props = $props();

	let inputAreaRef: ChatFormInputArea | undefined = $state(undefined);
	let message = $state(initialMessage);
	let previousIsLoading = $state(isLoading);
	let previousInitialMessage = $state(initialMessage);

	// Sync message when initialMessage prop changes (e.g., after draft restoration)
	$effect(() => {
		if (initialMessage !== previousInitialMessage) {
			message = initialMessage;
			previousInitialMessage = initialMessage;
		}
	});

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ message, files: uploadedFiles });
	}

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
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>

<ChatFormHelperText show={showHelperText} />
