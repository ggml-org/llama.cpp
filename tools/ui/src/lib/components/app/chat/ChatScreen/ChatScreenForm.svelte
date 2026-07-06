<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatForm } from '$lib/components/app';
	import { isMobile } from '$lib/stores/viewport.svelte';
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
		uploadedFiles = $bindable([])
	}: Props = $props();

	let chatFormRef: ChatForm | undefined = $state(undefined);
	let formWrapperEl: HTMLDivElement | undefined = $state();
	let chatId = $derived(page.params.id as string | undefined);

	$effect(() => {
		if (!formWrapperEl) return;

		const formEl = formWrapperEl.querySelector('form') as HTMLElement | null;
		if (!formEl) return;

		const updateHeight = () => {
			const height = Math.round(formEl.getBoundingClientRect().height);
			document.documentElement.style.setProperty('--chat-form-height', `${height}px`);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);
		resizeObserver.observe(formEl);

		return () => {
			resizeObserver.disconnect();
			document.documentElement.style.removeProperty('--chat-form-height');
		};
	});
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
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

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

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

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ message, files: uploadedFiles });
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	onMount(() => {
		if (!isMobile.current) {
			setTimeout(() => chatFormRef?.focus(), 100);
		}
	});

	afterNavigate((navigation) => {
		if (navigation?.from != null && !isMobile.current) {
			setTimeout(() => chatFormRef?.focus(), 100);
		}
	});

	$effect(() => {
		if (initialMessage !== previousInitialMessage) {
			message = initialMessage;
			previousInitialMessage = initialMessage;
		}
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(() => chatFormRef?.focus(), 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div class="chat-screen-form-wrapper" bind:this={formWrapperEl}>
	<ChatForm
		class="mx-auto max-w-3xl {className}"
		bind:this={chatFormRef}
		bind:value={message}
		bind:uploadedFiles
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
