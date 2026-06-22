<script lang="ts">
	import { page } from '$app/state';
	import {
		ChatScreenForm,
		ChatMessages,
		ChatScreenDragOverlay,
		ChatScreenProcessingInfo,
		ServerLoadingSplash,
		ChatScreenServerError
	} from '$lib/components/app';
	import { setProcessingInfoContext } from '$lib/contexts';
	import { createAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import { useChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
	import { useChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
	import { useChatScreenScroll } from '$lib/hooks/use-chat-screen-scroll.svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { device } from '$lib/stores/device.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import {
		chatStore,
		errorDialog,
		isLoading,
		isChatStreaming,
		isEditing,
		activeProcessingState
	} from '$lib/stores/chat.svelte';
	import {
		conversationsStore,
		activeMessages,
		activeConversation
	} from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { serverLoading, serverError } from '$lib/stores/server.svelte';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { onDestroy, onMount } from 'svelte';
	import ChatScreenGreeting from './ChatScreenGreeting.svelte';
	import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
	import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
	import { ROUTES } from '$lib/constants';

	let { showCenteredEmpty = false } = $props();

	const activeModel = useChatScreenActiveModel();

	const fileUpload = useChatScreenFileUpload({
		capabilities: () => ({
			hasVision: activeModel.hasVisionModality,
			hasAudio: activeModel.hasAudioModality,
			hasVideo: activeModel.hasVideoModality
		}),
		activeModelId: () => activeModel.activeModelId
	});

	const dragAndDrop = useChatScreenDragAndDrop({
		onDrop: fileUpload.handleFileUpload
	});

	let showEmptyFileDialog = $state(false);
	let emptyFileNames = $state<string[]>([]);
	let initialMessage = $state('');

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		const plainFiles = files ? $state.snapshot(files) : undefined;
		const result = plainFiles
			? await parseFilesToMessageExtras(plainFiles, activeModel.activeModelId ?? undefined)
			: undefined;

		if (result?.emptyFiles && result.emptyFiles.length > 0) {
			emptyFileNames = result.emptyFiles;
			showEmptyFileDialog = true;
			if (files) {
				const emptyFileNamesSet = new Set(result.emptyFiles);
				fileUpload.uploadedFiles = fileUpload.uploadedFiles.filter(
					(file) => !emptyFileNamesSet.has(file.name)
				);
			}
			return false;
		}

		if (!isMobile.current) {
			autoScroll.enable();
		}

		setTimeout(() => {
			scroll.chatScrollContainer?.scrollTo({
				top: scroll.chatScrollContainer.scrollHeight,
				behavior: 'smooth'
			});
		}, 100);

		if (isMobile.current) {
			autoScroll.setDisabled(disableAutoScroll);
		}

		await chatStore.sendMessage(message, result?.extras);
		autoScroll.setContainer(scroll.chatScrollContainer);
		return true;
	}

	async function handleSystemPromptAdd(draft: { message: string; files: ChatUploadedFile[] }) {
		if (draft.message || draft.files.length > 0) {
			chatStore.savePendingDraft(draft.message, draft.files);
		}
		await chatStore.addSystemPrompt();
	}

	onMount(() => {
		const pendingDraft = chatStore.consumePendingDraft();
		if (pendingDraft) {
			initialMessage = pendingDraft.message;
			fileUpload.uploadedFiles = pendingDraft.files;
		}
	});

	let showDeleteDialog = $state(false);

	async function handleDeleteConfirm() {
		const conversation = activeConversation();
		if (conversation) {
			await conversationsStore.deleteConversation(conversation.id);
		}
		showDeleteDialog = false;
	}

	const { handleKeydown } = useKeyboardShortcuts({
		deleteActiveConversation: () => {
			if (activeConversation()) {
				showDeleteDialog = true;
			}
		}
	});

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

	setProcessingInfoContext({
		get showProcessingInfo() {
			return showProcessingInfo;
		}
	});

	function handleErrorDialogOpenChange(open: boolean) {
		if (!open) {
			chatStore.dismissErrorDialog();
		}
	}

	const autoScroll = createAutoScrollController();
	const scroll = useChatScreenScroll(autoScroll);

	let disableAutoScroll = $derived(Boolean(config().disableAutoScroll) || isMobile.current);

	$effect(() => {
		if (!isMobile.current) {
			autoScroll.setDisabled(disableAutoScroll);
		}
	});

	// Disable auto-scroll while loading/streaming on desktop, re-enable on idle.
	$effect(() => {
		if (isMobile.current) return;
		const shouldDisableAutoScroll = config().disableAutoScroll || isCurrentConversationLoading;
		autoScroll.setDisabled(shouldDisableAutoScroll);
		if (!shouldDisableAutoScroll) {
			autoScroll.enable();
		}
	});

	onMount(() => {
		autoScroll.startObserving();
		if (!disableAutoScroll) {
			autoScroll.enable();
		}
	});

	onDestroy(() => autoScroll.destroy());
</script>

{#if dragAndDrop.isDragOver}
	<ChatScreenDragOverlay />
{/if}

<svelte:window onkeydown={handleKeydown} onscroll={scroll.handleScroll} />

{#if isServerLoading}
	<ServerLoadingSplash />
{:else}
	<div
		class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem)] md:min-h-full px-4 md:py-0 pt-12 pb-48 md:pb-4"
		ondragenter={dragAndDrop.dragHandlers.dragenter}
		ondragleave={dragAndDrop.dragHandlers.dragleave}
		ondragover={dragAndDrop.dragHandlers.dragover}
		ondrop={dragAndDrop.dragHandlers.drop}
		role="main"
	>
		{#if !isEmpty}
			<ChatMessages
				messages={activeMessages()}
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
				device.isStandalone
					? 'bottom-6 right-4 left-4'
					: device.isIOSSafari
						? 'bottom-1 left-2 right-2'
						: 'bottom-2 right-2 left-2',
				isEmpty
					? 'md:bottom-[calc(50dvh-7rem)] 2xl:bottom-[calc(50dvh-4rem)]'
					: 'md:bottom-4 pt-24 md:pt-24'
			]}
		>
			<ChatScreenGreeting {isEmpty} />

			<ChatScreenServerError />

			<ChatScreenProcessingInfo />

			{#if autoScroll.userScrolledUp && page.url.hash.includes(ROUTES.CHAT) && page.params.id}
				<ChatScreenActionScrollDown
					onclick={() => {
						scroll.chatScrollContainer?.scrollTo({
							top: scroll.chatScrollContainer.scrollHeight,
							behavior: 'smooth'
						});
					}}
				/>
			{/if}

			<ChatScreenForm
				class="pointer-events-auto conversation-chat-form"
				disabled={hasPropsError || isEditing()}
				{initialMessage}
				isLoading={isCurrentConversationLoading}
				onFileRemove={fileUpload.handleFileRemove}
				onFileUpload={fileUpload.handleFileUpload}
				onSend={handleSendMessage}
				onStop={() => chatStore.stopGeneration()}
				onSystemPromptAdd={handleSystemPromptAdd}
				bind:uploadedFiles={fileUpload.uploadedFiles}
			/>
		</div>
	</div>
{/if}

<ChatScreenDialogsAndAlerts
	showDeleteDialog={showDeleteDialog}
	handleDeleteConfirm={handleDeleteConfirm}
	showEmptyFileDialog={showEmptyFileDialog}
	emptyFileNames={emptyFileNames}
	activeErrorDialog={activeErrorDialog}
	handleErrorDialogOpenChange={handleErrorDialogOpenChange}
	fileUpload={fileUpload}
/>
