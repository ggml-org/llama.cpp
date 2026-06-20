<script lang="ts">
	import { Trash2 } from '@lucide/svelte';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ChatScreenForm,
		ChatMessages,
		ChatScreenDragOverlay,
		ChatScreenProcessingInfo,
		ChatScreenActionScrollDown,
		DialogEmptyFileAlert,
		DialogFileUploadError,
		DialogChatError,
		ServerLoadingSplash,
		DialogConfirmation,
		ChatScreenServerError
	} from '$lib/components/app';
	import { setProcessingInfoContext } from '$lib/contexts';
	import { ErrorDialogType } from '$lib/enums';
	import { createAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { device } from '$lib/stores/device.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import {
		chatStore,
		errorDialog,
		isLoading,
		isChatStreaming,
		isEditing,
		getAddFilesHandler,
		activeProcessingState
	} from '$lib/stores/chat.svelte';
	import {
		conversationsStore,
		activeMessages,
		activeConversation
	} from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { serverLoading, serverError, isRouterMode } from '$lib/stores/server.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isFileTypeSupported, filterFilesByModalities } from '$lib/utils';
	import { parseFilesToMessageExtras, processFilesToChatUploaded } from '$lib/utils/browser-only';
	import { onDestroy, onMount } from 'svelte';
	import ChatScreenGreeting from './ChatScreenGreeting.svelte';

	let { showCenteredEmpty = false } = $props();

	const autoScroll = createAutoScrollController();

	let disableAutoScroll = $derived(Boolean(config().disableAutoScroll) || isMobile.current);
	let chatScrollContainer: HTMLElement | undefined = $state();
	let dragCounter = $state(0);
	let isDragOver = $state(false);
	let showFileErrorDialog = $state(false);
	let uploadedFiles = $state<ChatUploadedFile[]>([]);

	let fileErrorData = $state<{
		generallyUnsupported: File[];
		modalityUnsupported: File[];
		modalityReasons: Record<string, string>;
		supportedTypes: string[];
	}>({
		generallyUnsupported: [],
		modalityUnsupported: [],
		modalityReasons: {},
		supportedTypes: []
	});

	let showDeleteDialog = $state(false);

	let showEmptyFileDialog = $state(false);

	let processingInfoVisible = $state(false);

	let emptyFileNames = $state<string[]>([]);

	let initialMessage = $state('');

	let isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

	let activeErrorDialog = $derived(errorDialog());
	let isServerLoading = $derived(serverLoading());
	let hasPropsError = $derived(!!serverError());

	let isCurrentConversationLoading = $derived(isLoading() || isChatStreaming());

	let showProcessingInfo = $derived(
		isCurrentConversationLoading ||
			(config().keepStatsVisible && !!page.params.id) ||
			activeProcessingState() !== null
	);

	let isRouter = $derived(isRouterMode());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0);

	setProcessingInfoContext({
		get showProcessingInfo() {
			return showProcessingInfo;
		}
	});

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

	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsAudio(activeModelId);
		}

		return false;
	});

	let hasVideoModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsVideo(activeModelId);
		}

		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsVision(activeModelId);
		}

		return false;
	});

	async function handleDeleteConfirm() {
		const conversation = activeConversation();

		if (conversation) {
			await conversationsStore.deleteConversation(conversation.id);
		}

		showDeleteDialog = false;
	}

	function handleProcessingInfoVisibility(visible: boolean) {
		processingInfoVisible = visible;
	}

	function handleDragEnter(event: DragEvent) {
		event.preventDefault();

		dragCounter++;

		if (event.dataTransfer?.types.includes('Files')) {
			isDragOver = true;
		}
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();

		dragCounter--;

		if (dragCounter === 0) {
			isDragOver = false;
		}
	}

	function handleErrorDialogOpenChange(open: boolean) {
		if (!open) {
			chatStore.dismissErrorDialog();
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();

		isDragOver = false;
		dragCounter = 0;

		if (event.dataTransfer?.files) {
			const files = Array.from(event.dataTransfer.files);

			if (isEditing()) {
				const handler = getAddFilesHandler();

				if (handler) {
					handler(files);
					return;
				}
			}

			processFiles(files);
		}
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	function handleFileUpload(files: File[]) {
		processFiles(files);
	}

	const { handleKeydown } = useKeyboardShortcuts({
		deleteActiveConversation: () => {
			if (activeConversation()) {
				showDeleteDialog = true;
			}
		}
	});

	async function handleSystemPromptAdd(draft: { message: string; files: ChatUploadedFile[] }) {
		if (draft.message || draft.files.length > 0) {
			chatStore.savePendingDraft(draft.message, draft.files);
		}

		await chatStore.addSystemPrompt();
	}

	function handleScroll() {
		autoScroll.handleScroll();
	}

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		const plainFiles = files ? $state.snapshot(files) : undefined;
		const result = plainFiles
			? await parseFilesToMessageExtras(plainFiles, activeModelId ?? undefined)
			: undefined;

		if (result?.emptyFiles && result.emptyFiles.length > 0) {
			emptyFileNames = result.emptyFiles;
			showEmptyFileDialog = true;

			if (files) {
				const emptyFileNamesSet = new Set(result.emptyFiles);
				uploadedFiles = uploadedFiles.filter((file) => !emptyFileNamesSet.has(file.name));
			}
			return false;
		}

		const extras = result?.extras;

		// Enable autoscroll for user-initiated message sending
		autoScroll.enable();
		await chatStore.sendMessage(message, extras);
		autoScroll.scrollToBottom();

		return true;
	}

	async function processFiles(files: File[]) {
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		// Use model-specific capabilities for file validation
		const capabilities = {
			hasVision: hasVisionModality,
			hasAudio: hasAudioModality,
			hasVideo: hasVideoModality
		};
		const { supportedFiles, unsupportedFiles, modalityReasons } = filterFilesByModalities(
			generallySupported,
			capabilities
		);

		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			const supportedTypes: string[] = ['text files', 'PDFs'];

			if (hasVisionModality) supportedTypes.push('images');
			if (hasAudioModality) supportedTypes.push('audio files');
			if (hasVideoModality) supportedTypes.push('video files');

			fileErrorData = {
				generallyUnsupported,
				modalityUnsupported: unsupportedFiles,
				modalityReasons,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const processed = await processFilesToChatUploaded(
				supportedFiles,
				activeModelId ?? undefined
			);
			uploadedFiles = [...uploadedFiles, ...processed];
		}
	}

	afterNavigate(() => {
		if (!disableAutoScroll) {
			autoScroll.enable();
		}
	});

	function handleMessagesReady() {
		// Respect the user's explicit setting, but keep the one-time scroll on
		// the messages-ready $effect even on mobile (where the auto-scroll
		// controller is disabled). Unlike the per-token auto-scroll, this only
		// fires when allConversationMessages changes (load, edit, regenerate, ...).
		if (config().disableAutoScroll) return;

		if (!autoScroll.userScrolledUp) {
			requestAnimationFrame(() => {
				chatScrollContainer?.scrollTo({
					top: chatScrollContainer.scrollHeight,
					behavior: 'instant'
				});
			});
		}
	}

	onMount(() => {
		autoScroll.startObserving();

		if (!disableAutoScroll) {
			autoScroll.enable();
		}

		const pendingDraft = chatStore.consumePendingDraft();
		if (pendingDraft) {
			initialMessage = pendingDraft.message;
			uploadedFiles = pendingDraft.files;
		}
	});

	onDestroy(() => autoScroll.destroy());

	$effect(() => {
		chatScrollContainer = document.documentElement;
		autoScroll.setContainer(chatScrollContainer);
	});

	$effect(() => {
		autoScroll.setDisabled(disableAutoScroll);
	});
</script>

{#if isDragOver}
	<ChatScreenDragOverlay />
{/if}

<svelte:window onkeydown={handleKeydown} onscroll={handleScroll} />

{#if isServerLoading}
	<ServerLoadingSplash />
{:else}
	<div
		class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem)] md:min-h-full px-2 md:px-4 md:py-0 pt-12 pb-56 md:pb-4"
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		role="main"
	>
		{#if !isEmpty}
			<ChatMessages
				messages={activeMessages()}
				onMessagesReady={handleMessagesReady}
				onUserAction={() => {
					autoScroll.enable();
					if (!autoScroll.userScrolledUp) {
						autoScroll.scrollToBottom();
					}
				}}
			/>
		{/if}

		<div
			class={[
				'pointer-events-none md:sticky fixed  mt-auto transition-all duration-200',
				device.isStandalone ? 'bottom-6 right-4 left-4' : device.isIOSSafari ? 'bottom-1 left-2 right-2' : 'bottom-2 right-2 left-2',
				isEmpty ? 'md:bottom-[calc(50dvh-4rem)]' : 'md:bottom-4 pt-24 md:pt-32'
			]}
		>
			<ChatScreenGreeting {isEmpty} />

			<!-- <ChatScreenActionScrollDown
				container={chatScrollContainer}
				hasProcessingInfoVisible={processingInfoVisible}
			/> -->

			<ChatScreenServerError />

			<ChatScreenProcessingInfo onVisibilityChange={handleProcessingInfoVisibility} />

			<div class="conversation-chat-form pointer-events-auto rounded-t-3xl">
				<ChatScreenForm
					disabled={hasPropsError || isEditing()}
					{initialMessage}
					isLoading={isCurrentConversationLoading}
					onFileRemove={handleFileRemove}
					onFileUpload={handleFileUpload}
					onSend={handleSendMessage}
					onStop={() => chatStore.stopGeneration()}
					onSystemPromptAdd={handleSystemPromptAdd}
					bind:uploadedFiles
				/>
			</div>
		</div>
	</div>
{/if}

<DialogFileUploadError bind:open={showFileErrorDialog} {fileErrorData} />

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete Conversation"
	description="Are you sure you want to delete this conversation? This action cannot be undone and will permanently remove all messages in this conversation."
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteConfirm}
	onCancel={() => (showDeleteDialog = false)}
/>

<DialogEmptyFileAlert
	bind:open={showEmptyFileDialog}
	emptyFiles={emptyFileNames}
	onOpenChange={(open) => {
		if (!open) {
			emptyFileNames = [];
		}
	}}
/>

<DialogChatError
	message={activeErrorDialog?.message ?? ''}
	contextInfo={activeErrorDialog?.contextInfo}
	onOpenChange={handleErrorDialogOpenChange}
	open={Boolean(activeErrorDialog)}
	type={activeErrorDialog?.type ?? ErrorDialogType.SERVER}
/>
