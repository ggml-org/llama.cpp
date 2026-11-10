<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatForm } from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { useDraftMessages } from '$lib/hooks/use-draft-messages.svelte';
	import { deviceStore, settingsStore } from '$lib/stores';
	import { hapticTap } from '$lib/utils';
	import { onMount, tick } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
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
		if ((!message.trim() && uploadedFiles.length === 0) || disabled || hasLoadingAttachments)
			return;

		if (!chatFormRef?.checkModelSelected()) return;

		// the message goes: a phone confirms it by feel. Before any await, so it still runs inside
		// the tap or keypress that sent it, which iOS needs to play the tick
		hapticTap();

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];
		clearDraft();

		// called in the same task: the page still sees the form as it was sent (expanded or not)
		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
			await tick();
			chatFormRef?.expandForMessage();
		}
	}

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ files: uploadedFiles, message });
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	export function openWorkingDirectoryPicker() {
		chatFormRef?.openWorkingDirectoryPicker();
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
		expandable
		{isLoading}
		onFilesAdd={handleFilesAdd}
		{onStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
		singleRow={settingsStore.config[SETTINGS_KEYS.SINGLE_ROW_CHAT_FORM] !== false}
	/>
</div>
