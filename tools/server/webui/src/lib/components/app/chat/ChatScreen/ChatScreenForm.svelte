<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatFormHelperText, ChatForm } from '$lib/components/app';
	import { onMount } from 'svelte';
	import { useDraftMessages } from '$lib/hooks/use-draft-messages.svelte';

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

	let chatFormRef: ChatForm | undefined = $state(undefined);
	let chatId = $derived(page.params.id as string | undefined);
	let message = $derived(initialMessage);
	let previousIsLoading = $derived(isLoading);
	let previousInitialMessage = $derived(initialMessage);

	const { clearDraft } = useDraftMessages({
		getChatId: () => chatId,
		getMessage: () => message,
		getFiles: () => uploadedFiles,
		setMessage: (m) => (message = m),
		setFiles: (f) => (uploadedFiles = f),
		getInitialMessage: () => initialMessage
	});

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
		if ((!message.trim() && uploadedFiles.length === 0) || disabled || hasLoadingAttachments)
			return;

		if (!chatFormRef?.checkModelSelected()) return;

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];
		clearDraft();

		chatFormRef?.resetTextareaHeight();

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
		setTimeout(() => chatFormRef?.focus(), 10);
	});

	afterNavigate((navigation) => {
		if (navigation?.from != null) {
			setTimeout(() => chatFormRef?.focus(), 10);
		}
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(() => chatFormRef?.focus(), 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div class="relative mx-auto max-w-[48rem]">
	<ChatForm
		bind:this={chatFormRef}
		bind:value={message}
		bind:uploadedFiles
		class={className}
		{disabled}
		{isLoading}
		showMcpPromptButton
		onFilesAdd={handleFilesAdd}
		{onStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>

<ChatFormHelperText show={showHelperText} />
