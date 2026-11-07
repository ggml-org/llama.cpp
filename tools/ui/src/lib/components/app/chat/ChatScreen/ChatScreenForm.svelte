<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatForm } from '$lib/components/app';
	import { useDraftMessages } from '$lib/hooks/use-draft-messages.svelte';
	import { deviceStore } from '$lib/stores';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (
			message: string,
			files: ChatUploadedFile[],
			onPrepared: (emptyFiles: string[]) => boolean
		) => Promise<void>;
		onStop?: () => void;
		onSystemPromptAdd?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
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
	let message = $state('');
	let isPreparingSend = false;
	let previousIsLoading = $derived(isLoading);

	const { clearDraft } = useDraftMessages({
		getChatId: () => chatId,
		getFiles: () => uploadedFiles,
		getMessage: () => message,
		setFiles: (f) => (uploadedFiles = f),
		setMessage: (m) => (message = m)
	});

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

	async function handleSubmit() {
		if (
			(!message.trim() && uploadedFiles.length === 0) ||
			disabled ||
			hasLoadingAttachments ||
			isPreparingSend
		)
			return;

		if (!chatFormRef?.checkModelSelected()) return;

		const originalMessage = message;
		const originalChatId = chatId;
		const filesToSend = [...uploadedFiles];

		let preparationPending = true;

		isPreparingSend = true;

		try {
			await onSend?.(originalMessage.trim(), filesToSend, (emptyFiles) => {
				preparationPending = false;
				isPreparingSend = false;

				// Cancel stale preparation so it cannot clear or restore over a newer draft.
				if (
					chatId !== originalChatId ||
					message !== originalMessage ||
					uploadedFiles.length !== filesToSend.length ||
					uploadedFiles.some((file, index) => file !== filesToSend[index])
				) {
					toast.info('Message changed while preparing attachments. Send again.');

					return false;
				}

				if (emptyFiles.length > 0) {
					uploadedFiles = uploadedFiles.filter((file) => !emptyFiles.includes(file.name));

					return true;
				}

				message = '';
				uploadedFiles = [];
				clearDraft();
				chatFormRef?.resetTextareaHeight();

				return true;
			});
		} finally {
			// Once accepted, another submission may prepare while this reply streams.
			if (preparationPending) isPreparingSend = false;
		}
	}

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ files: uploadedFiles, message });
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	// Auto-focus must not steal focus already claimed elsewhere (e.g. the system
	// message editor opened just before a navigation)
	function focusFormUnlessCaptured() {
		const active = document.activeElement;

		if (active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement) return;

		chatFormRef?.focus();
	}

	onMount(() => {
		if (!deviceStore.isMobile) {
			setTimeout(focusFormUnlessCaptured, 100);
		}
	});

	afterNavigate((navigation) => {
		if (navigation?.from != null && !deviceStore.isMobile) {
			setTimeout(focusFormUnlessCaptured, 100);
		}
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(focusFormUnlessCaptured, 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div bind:this={formWrapperEl} class="chat-screen-form-wrapper">
	<ChatForm
		bind:this={chatFormRef}
		bind:uploadedFiles
		bind:value={message}
		class="mx-auto max-w-3xl {className}"
		{disabled}
		{isLoading}
		onFilesAdd={handleFilesAdd}
		{onStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>
