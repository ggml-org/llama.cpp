<script lang="ts">
	import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';
	import { processFilesToChatUploaded } from '$lib/utils/process-uploaded-files';
	import { serverStore } from '$lib/stores/server.svelte';
	import { isFileTypeSupported } from '$lib/constants/supported-file-types';
	import { filterFilesByModalities } from '$lib/utils/modality-file-validation';
	import { supportsVision, supportsAudio, serverError, serverLoading } from '$lib/stores/server.svelte';
	import { ChatForm, ChatScreenHeader, ChatMessages, ServerInfo, ServerErrorSplash, ServerLoadingSplash } from '$lib/components/app';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration,
		setMaxContextError
	} from '$lib/stores/chat.svelte';
	import { contextService } from '$lib/services/context';
	import { fade, fly, slide } from 'svelte/transition';
	import { AUTO_SCROLL_THRESHOLD } from '$lib/constants/auto-scroll';
	import { navigating } from '$app/state';
	import ChatScreenDragOverlay from './ChatScreenDragOverlay.svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { deleteConversation } from '$lib/stores/chat.svelte';
	import { goto } from '$app/navigation';


	let { showCenteredEmpty = false } = $props();
	let chatScrollContainer: HTMLDivElement | undefined = $state();
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let autoScrollEnabled = $state(true);
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let isDragOver = $state(false);
	let dragCounter = $state(0);

	// Alert Dialog state for file upload errors
	let showFileErrorDialog = $state(false);
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

	const isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

	const hasServerError = $derived(serverError());
	const isServerLoading = $derived(serverLoading());

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

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		dragCounter = 0;

		if (event.dataTransfer?.files) {
			processFiles(Array.from(event.dataTransfer.files));
		}
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	function handleFileUpload(files: File[]) {
		processFiles(files);
	}

	function handleScroll() {
		if (!chatScrollContainer) return;

		const { scrollTop, scrollHeight, clientHeight } = chatScrollContainer;
		const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

		if (distanceFromBottom > AUTO_SCROLL_THRESHOLD) {
			autoScrollEnabled = false;
		} else if (distanceFromBottom <= AUTO_SCROLL_THRESHOLD) {
			autoScrollEnabled = true;
		}
	}

	async function handleSendMessage(
		message: string,
		files?: ChatUploadedFile[]
	): Promise<boolean> {
		const extras = files ? await parseFilesToMessageExtras(files) : undefined;

		// Check context limit using real-time slots data
		const contextCheck = await contextService.checkContextLimit();

		if (contextCheck && contextCheck.wouldExceed) {
			const errorMessage = contextService.getContextErrorMessage(contextCheck);

			setMaxContextError({
				message: errorMessage,
				estimatedTokens: contextCheck.currentUsage,
				maxContext: contextCheck.maxContext
			});

			return false;
		}

		await sendMessage(message, extras);
		scrollChatToBottom();

		return true;
	}

	async function processFiles(files: File[]) {
		// First filter by general file type support
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		// Then filter by model modalities
		const { supportedFiles, unsupportedFiles, modalityReasons } =
			filterFilesByModalities(generallySupported);

		// Combine all unsupported files
		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			// Determine supported types for current model
			const supportedTypes: string[] = ['text files', 'PDFs'];
			if (supportsVision()) supportedTypes.push('images');
			if (supportsAudio()) supportedTypes.push('audio files');

			// Structure error data for better presentation
			fileErrorData = {
				generallyUnsupported,
				modalityUnsupported: unsupportedFiles,
				modalityReasons,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const processed = await processFilesToChatUploaded(supportedFiles);
			uploadedFiles = [...uploadedFiles, ...processed];
		}
	}

	function scrollChatToBottom(behavior: ScrollBehavior = 'smooth') {
		chatScrollContainer?.scrollTo({
			top: chatScrollContainer?.scrollHeight,
			behavior
		});
	}

	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.key === 'k') {
			event.preventDefault();
			goto('/?new_chat=true');
		}
		
		if (isCtrlOrCmd && event.shiftKey && (event.key === 'd' || event.key === 'D')) {
			event.preventDefault();
			if (activeConversation()) {
				showDeleteDialog = true;
			}
		}
	}

	async function handleDeleteConfirm() {
		const conversation = activeConversation();
		if (conversation) {
			await deleteConversation(conversation.id);
		}
		showDeleteDialog = false;
	}

	$effect(() => {
		// This solution is not ideal, but it works for now. But can be tricky for long conversations
		// Eventually we might want to find a proper way to render the content scrolled down from the beginning
		if (navigating.complete && chatScrollContainer) {
			setTimeout(() => scrollChatToBottom('instant'), 100);
		}

		if (navigating) {
			scrollChatToBottom('instant');
		}
	});

	$effect(() => {
		if (isLoading() && autoScrollEnabled) {
			scrollInterval = setInterval(scrollChatToBottom, 50);
		} else if (scrollInterval) {
			clearInterval(scrollInterval);
			scrollInterval = undefined;
		}
	});
</script>

{#if isDragOver}
	<ChatScreenDragOverlay />
{/if}

<svelte:window onkeydown={handleKeydown} />

<ChatScreenHeader />

{#if !isEmpty}
	<div
		class="flex h-full flex-col overflow-y-auto px-4 md:px-6"
		bind:this={chatScrollContainer}
		onscroll={handleScroll}
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		role="main"
		aria-label="Chat interface with file drop zone"
	>
		<ChatMessages class="mb-16 md:mb-24" messages={activeMessages()} />

		<div class="sticky bottom-0 left-0 right-0 mt-auto" in:slide={{ duration: 150, axis: 'y' }}>
			<div class="conversation-chat-form rounded-t-3xl pb-4">
				<ChatForm
					isLoading={isLoading()}
					onFileRemove={handleFileRemove}
					onFileUpload={handleFileUpload}
					onSend={handleSendMessage}
					onStop={() => stopGeneration()}
					showHelperText={false}
					bind:uploadedFiles
				/>
			</div>
		</div>
	</div>
{:else if hasServerError}
	<!-- Server Error State -->
	<ServerErrorSplash error={hasServerError} />
{:else if isServerLoading}
	<!-- Server Loading State -->
	<ServerLoadingSplash />
{:else if serverStore.modelName}
	<div
		aria-label="Welcome screen with file drop zone"
		class="flex h-full items-center justify-center"
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		role="main"
	>
		<div class="w-full max-w-2xl px-4">
			<div class="mb-8 text-center" in:fade={{ duration: 300 }}>
				<h1 class="mb-2 text-3xl font-semibold tracking-tight">llama.cpp</h1>

				<p class="text-muted-foreground text-lg">How can I help you today?</p>
			</div>

			<div class="mb-6 flex justify-center" in:fly={{ y: 10, duration: 300, delay: 200 }}>
				<ServerInfo />
			</div>

			<div in:fly={{ y: 10, duration: 250, delay: 300 }}>
				<ChatForm
					isLoading={isLoading()}
					onFileRemove={handleFileRemove}
					onFileUpload={handleFileUpload}
					onSend={handleSendMessage}
					onStop={() => stopGeneration()}
					showHelperText={true}
					bind:uploadedFiles
				/>
			</div>
		</div>
	</div>
{/if}

<!-- File Upload Error Alert Dialog -->
<AlertDialog.Root bind:open={showFileErrorDialog}>
	<AlertDialog.Portal>
		<AlertDialog.Overlay />
		<AlertDialog.Content class="max-w-md">
			<AlertDialog.Header>
				<AlertDialog.Title>File Upload Error</AlertDialog.Title>
				<AlertDialog.Description class="text-muted-foreground text-sm">
					Some files cannot be uploaded with the current model.
				</AlertDialog.Description>
			</AlertDialog.Header>

			<div class="space-y-4">
				<!-- Generally unsupported files -->
				{#if fileErrorData.generallyUnsupported.length > 0}
					<div class="space-y-2">
						<h4 class="text-destructive text-sm font-medium">Unsupported File Types</h4>
						<div class="space-y-1">
							{#each fileErrorData.generallyUnsupported as file}
								<div class="bg-destructive/10 rounded-md px-3 py-2">
									<p class="text-destructive break-all font-mono text-sm">
										{file.name}
									</p>
									<p class="text-muted-foreground mt-1 text-xs">
										File type not supported
									</p>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Modality-restricted files -->
				{#if fileErrorData.modalityUnsupported.length > 0}
					<div class="space-y-2">
						<h4 class="text-destructive text-sm font-medium">
							Model Compatibility Issues
						</h4>
						<div class="space-y-1">
							{#each fileErrorData.modalityUnsupported as file}
								<div class="bg-destructive/10 rounded-md px-3 py-2">
									<p class="text-destructive break-all font-mono text-sm">
										{file.name}
									</p>
									<p class="text-muted-foreground mt-1 text-xs">
										{fileErrorData.modalityReasons[file.name] ||
											'Not supported by current model'}
									</p>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Supported file types -->
				<div class="bg-muted/50 rounded-md p-3">
					<h4 class="mb-2 text-sm font-medium">This model supports:</h4>
					<p class="text-muted-foreground text-sm">
						{fileErrorData.supportedTypes.join(', ')}
					</p>
				</div>
			</div>

			<AlertDialog.Footer>
				<AlertDialog.Action onclick={() => (showFileErrorDialog = false)}>
					Got it
				</AlertDialog.Action>
			</AlertDialog.Footer>
		</AlertDialog.Content>
	</AlertDialog.Portal>
</AlertDialog.Root>

<!-- Delete Chat Confirmation Dialog -->
<AlertDialog.Root bind:open={showDeleteDialog}>
	<AlertDialog.Portal>
		<AlertDialog.Overlay />
		<AlertDialog.Content class="max-w-md">
			<AlertDialog.Header>
				<AlertDialog.Title>Delete Chat</AlertDialog.Title>
				<AlertDialog.Description class="text-muted-foreground text-sm">
					Are you sure you want to delete this chat? This action cannot be undone.
				</AlertDialog.Description>
			</AlertDialog.Header>

			<AlertDialog.Footer>
				<AlertDialog.Cancel onclick={() => (showDeleteDialog = false)}>
					Cancel
				</AlertDialog.Cancel>
				<AlertDialog.Action onclick={handleDeleteConfirm} class="bg-destructive text-destructive-foreground hover:bg-destructive/90">
					Delete
				</AlertDialog.Action>
			</AlertDialog.Footer>
		</AlertDialog.Content>
	</AlertDialog.Portal>
</AlertDialog.Root>

<style>
	.conversation-chat-form {
		position: relative;

		&::after {
			content: '';
			position: fixed;
			bottom: 0;
			z-index: -1;
			left: 0;
			right: 0;
			width: 100%;
			height: 2.375rem;
			background-color: var(--background);
		}
	}
</style>
