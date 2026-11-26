<script lang="ts">
	import { ChatForm } from '#lib/components/app/index.ts';
	import { SETTINGS_KEYS, SYSTEM_MESSAGE_PLACEHOLDER } from '#lib/constants/index.ts';
	import { MessageRole } from '#lib/enums/index.ts';
	import { useDraftMessages } from '#lib/hooks/use-draft-messages.svelte.ts';
	import { chatStore, conversationsStore, deviceStore, settingsStore } from '#lib/stores/index.ts';
	import type { FileProcessingResult } from '#lib/types/index.ts';
	import { reportAttachmentReadFailures } from '#lib/utils/browser-only.ts';
	import { hapticTap, shouldTypeToFocus } from '#lib/utils/index.ts';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { onDestroy, onMount, tick } from 'svelte';
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
			accept: (result?: FileProcessingResult) => boolean
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
	let pendingFiles: ChatUploadedFile[] | undefined;
	let previousIsLoading = $derived(isLoading);

	let isSystemMode = $state(false);
	let stashedDraft = $state<{ files: ChatUploadedFile[]; message: string } | null>(null);

	const { clearDraft } = useDraftMessages({
		getChatId: () => chatId,
		getFiles: () => (isSystemMode ? (stashedDraft?.files ?? []) : uploadedFiles),
		getMessage: () => (isSystemMode ? (stashedDraft?.message ?? '') : message),
		setFiles: (f) => (uploadedFiles = f),
		setMessage: (m) => (message = m)
	});

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

	async function handleSubmit() {
		if (isSystemMode) return handleSystemSubmit();

		if (
			(!message.trim() && uploadedFiles.length === 0) ||
			disabled ||
			hasLoadingAttachments ||
			pendingFiles
		)
			return;

		if (!chatFormRef?.checkModelSelected()) return;

		// the message goes: a phone confirms it by feel. Before any await, so it still runs inside
		// the tap or keypress that sent it, which iOS needs to play the tick
		hapticTap();

		const originalMessage = message;
		const originalChatId = chatId;
		const files = [...uploadedFiles];

		pendingFiles = files;

		try {
			await onSend?.(message.trim(), files, (result) => {
				if (pendingFiles !== files) return false;

				// Release preparation before streaming so the user can compose the next draft.
				pendingFiles = undefined;

				if (disabled || isSystemMode || chatId !== originalChatId) return false;

				if (result?.failedFiles.length) {
					reportAttachmentReadFailures(result.failedFiles);

					return false;
				}

				if (
					message !== originalMessage ||
					files.length !== uploadedFiles.length ||
					files.some((file, index) => file !== uploadedFiles[index])
				) {
					toast.info('Message changed while preparing attachments. Send again.');

					return false;
				}

				if (result?.emptyFiles.length) {
					const emptyIds = new Set(result.emptyFiles.map((file) => file.id));

					uploadedFiles = uploadedFiles.filter((file) => !emptyIds.has(file.id));
				} else {
					message = '';
					uploadedFiles = [];
					clearDraft();
				}

				return true;
			});
		} finally {
			if (pendingFiles === files) pendingFiles = undefined;
		}
	}

	// With system messages hidden in the conversation there is nothing to edit in place, so the
	// form itself turns into the system message editor

	function handleSystemPromptClick() {
		pendingFiles = undefined;

		if (settingsStore.config[SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE] ?? true) {
			onSystemPromptAdd?.({ files: uploadedFiles, message });

			return;
		}

		const existing = conversationsStore.activeMessages.find((m) => m.role === MessageRole.SYSTEM);

		stashedDraft = { files: uploadedFiles, message };
		const current = conversationsStore.activeConversation
			? existing?.content
			: chatStore.pendingSystemMessage;

		message = current && current !== SYSTEM_MESSAGE_PLACEHOLDER ? current : '';
		uploadedFiles = [];
		isSystemMode = true;
		void tick().then(() => chatFormRef?.focus());
	}

	function exitSystemMode(restoreDraft: boolean) {
		isSystemMode = false;

		if (restoreDraft && stashedDraft) {
			message = stashedDraft.message;
			uploadedFiles = stashedDraft.files;
		}

		stashedDraft = null;
		void tick().then(() => chatFormRef?.focus());
	}

	async function handleSystemSubmit() {
		const content = message.trim();

		if (!content) return;

		const draft = stashedDraft ?? { files: [], message: '' };

		isSystemMode = false;
		stashedDraft = null;
		message = draft.message;
		uploadedFiles = draft.files;

		await chatStore.saveSystemPrompt(content);
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

	// Typing while nothing takes text starts the message: focused during keydown, the
	// character the key types lands in the form
	function handleWindowKeydown(event: KeyboardEvent) {
		if (isSystemMode && event.key === 'Escape') {
			exitSystemMode(true);

			return;
		}

		if (disabled || deviceStore.isMobile || !shouldTypeToFocus(event)) return;

		chatFormRef?.focus();
	}

	onMount(() => {
		if (!deviceStore.isMobile) {
			setTimeout(focusFormUnlessCaptured, 100);
		}
	});

	onDestroy(() => {
		pendingFiles = undefined;
	});

	afterNavigate((navigation) => {
		if (navigation?.from == null) return;

		pendingFiles = undefined;

		if (!deviceStore.isMobile) {
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

<svelte:window onkeydown={handleWindowKeydown} />

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
		placeholder={isSystemMode ? 'System message...' : undefined}
		singleRow={settingsStore.config[SETTINGS_KEYS.SINGLE_ROW_CHAT_FORM] !== false}
		systemMode={isSystemMode}
	/>
</div>
