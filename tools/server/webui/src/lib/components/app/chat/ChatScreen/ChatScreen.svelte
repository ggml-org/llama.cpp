<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import {
		ChatForm,
		ChatScreenHeader,
		ChatScreenWarning,
		ChatMessages,
		ChatProcessingInfo,
		EmptyFileAlertDialog,
		ChatErrorDialog,
		ServerErrorSplash,
		ServerInfo,
		ServerLoadingSplash,
		ConfirmationDialog
	} from '$lib/components/app';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import {
		AUTO_SCROLL_AT_BOTTOM_THRESHOLD,
		AUTO_SCROLL_INTERVAL,
		INITIAL_SCROLL_DELAY
	} from '$lib/constants/auto-scroll';
	import {
		activeMessages,
		activeConversation,
		deleteConversation,
		dismissErrorDialog,
		errorDialog,
		isLoading,
		sendMessage,
		stopGeneration
	} from '$lib/stores/chat.svelte';
	import {
		supportsVision,
		supportsAudio,
		serverLoading,
		serverWarning,
		serverStore
	} from '$lib/stores/server.svelte';
	import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';
	import { isFileTypeSupported } from '$lib/utils/file-type';
	import { filterFilesByModalities } from '$lib/utils/modality-file-validation';
	import { processFilesToChatUploaded } from '$lib/utils/process-uploaded-files';
	import { onMount } from 'svelte';
	import { fade, fly, slide } from 'svelte/transition';
	import { Trash2 } from '@lucide/svelte';
	import ChatScreenDragOverlay from './ChatScreenDragOverlay.svelte';

	let { showCenteredEmpty = false } = $props();

	let autoScrollEnabled = $state(true);
	let chatScrollContainer: HTMLDivElement | undefined = $state();
	let dragCounter = $state(0);
	let isDragOver = $state(false);
	let lastScrollTop = $state(0);
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let scrollTimeout: ReturnType<typeof setTimeout> | undefined;
	let showFileErrorDialog = $state(false);
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let userScrolledUp = $state(false);

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

	let emptyFileNames = $state<string[]>([]);

	let isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

	let activeErrorDialog = $derived(errorDialog());
	let isServerLoading = $derived(serverLoading());

	let isCurrentConversationLoading = $derived(isLoading());

	async function handleDeleteConfirm() {
		const conversation = activeConversation();
		if (conversation) {
			await deleteConversation(conversation.id);
		}
		showDeleteDialog = false;
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
			dismissErrorDialog();
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

	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.shiftKey && (event.key === 'd' || event.key === 'D')) {
			event.preventDefault();
			if (activeConversation()) {
				showDeleteDialog = true;
			}
		}
	}

	function handleScroll() {
		if (!chatScrollContainer) return;

		const { scrollTop, scrollHeight, clientHeight } = chatScrollContainer;
		const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
		const isAtBottom = distanceFromBottom < AUTO_SCROLL_AT_BOTTOM_THRESHOLD;

		if (scrollTop < lastScrollTop && !isAtBottom) {
			userScrolledUp = true;
			autoScrollEnabled = false;
		} else if (isAtBottom && userScrolledUp) {
			userScrolledUp = false;
			autoScrollEnabled = true;
		}

		if (scrollTimeout) {
			clearTimeout(scrollTimeout);
		}

		scrollTimeout = setTimeout(() => {
			if (isAtBottom) {
				userScrolledUp = false;
				autoScrollEnabled = true;
			}
		}, AUTO_SCROLL_INTERVAL);

		lastScrollTop = scrollTop;
	}

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		const result = files ? await parseFilesToMessageExtras(files) : undefined;

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
		userScrolledUp = false;
		autoScrollEnabled = true;
		await sendMessage(message, extras);
		scrollChatToBottom();

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

		const { supportedFiles, unsupportedFiles, modalityReasons } =
			filterFilesByModalities(generallySupported);

		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			const supportedTypes: string[] = ['text files', 'PDFs'];

			if (supportsVision()) supportedTypes.push('images');
			if (supportsAudio()) supportedTypes.push('audio files');

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

	afterNavigate(() => {
		setTimeout(() => scrollChatToBottom('instant'), INITIAL_SCROLL_DELAY);
	});

	onMount(() => {
		setTimeout(() => scrollChatToBottom('instant'), INITIAL_SCROLL_DELAY);
	});

	$effect(() => {
		if (isCurrentConversationLoading && autoScrollEnabled) {
			scrollInterval = setInterval(scrollChatToBottom, AUTO_SCROLL_INTERVAL);
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
		bind:this={chatScrollContainer}
		aria-label="Chat interface with file drop zone"
		class="flex h-full flex-col overflow-y-auto px-4 md:px-6"
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		onscroll={handleScroll}
		role="main"
	>
		<ChatMessages
			class="mb-16 md:mb-24"
			messages={activeMessages()}
			onUserAction={() => {
				userScrolledUp = false;
				autoScrollEnabled = true;
				scrollChatToBottom();
			}}
		/>

		<div
			class="pointer-events-none sticky right-0 bottom-0 left-0 mt-auto"
			in:slide={{ duration: 150, axis: 'y' }}
		>
			<ChatProcessingInfo />

			{#if serverWarning()}
				<ChatScreenWarning class="pointer-events-auto mx-auto max-w-[48rem] px-4" />
			{/if}

			<div class="conversation-chat-form pointer-events-auto rounded-t-3xl pb-4">
				<ChatForm
					isLoading={isCurrentConversationLoading}
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
{:else if isServerLoading}
	<!-- Server Loading State -->
	<ServerLoadingSplash />
{:else if serverStore.error && !serverStore.modelName}
	<ServerErrorSplash error={serverStore.error} />
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

				<p class="text-lg text-muted-foreground">How can I help you today?</p>
			</div>

			<div class="mb-6 flex justify-center" in:fly={{ y: 10, duration: 300, delay: 200 }}>
				<ServerInfo />
			</div>

			{#if serverWarning()}
				<ChatScreenWarning />
			{/if}

			<div in:fly={{ y: 10, duration: 250, delay: 300 }}>
				<ChatForm
					isLoading={isCurrentConversationLoading}
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

				<AlertDialog.Description class="text-sm text-muted-foreground">
					Some files cannot be uploaded with the current model.
				</AlertDialog.Description>
			</AlertDialog.Header>

			<div class="space-y-4">
				{#if fileErrorData.generallyUnsupported.length > 0}
					<div class="space-y-2">
						<h4 class="text-sm font-medium text-destructive">Unsupported File Types</h4>

						<div class="space-y-1">
							{#each fileErrorData.generallyUnsupported as file (file.name)}
								<div class="rounded-md bg-destructive/10 px-3 py-2">
									<p class="font-mono text-sm break-all text-destructive">
										{file.name}
									</p>

									<p class="mt-1 text-xs text-muted-foreground">File type not supported</p>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				{#if fileErrorData.modalityUnsupported.length > 0}
					<div class="space-y-2">
						<h4 class="text-sm font-medium text-destructive">Model Compatibility Issues</h4>

						<div class="space-y-1">
							{#each fileErrorData.modalityUnsupported as file (file.name)}
								<div class="rounded-md bg-destructive/10 px-3 py-2">
									<p class="font-mono text-sm break-all text-destructive">
										{file.name}
									</p>

									<p class="mt-1 text-xs text-muted-foreground">
										{fileErrorData.modalityReasons[file.name] || 'Not supported by current model'}
									</p>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<div class="rounded-md bg-muted/50 p-3">
					<h4 class="mb-2 text-sm font-medium">This model supports:</h4>

					<p class="text-sm text-muted-foreground">
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

<ConfirmationDialog
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

<EmptyFileAlertDialog
	bind:open={showEmptyFileDialog}
	emptyFiles={emptyFileNames}
	onOpenChange={(open) => {
		if (!open) {
			emptyFileNames = [];
		}
	}}
/>

<ChatErrorDialog
	message={activeErrorDialog?.message ?? ''}
	onOpenChange={handleErrorDialogOpenChange}
	open={Boolean(activeErrorDialog)}
	type={activeErrorDialog?.type ?? 'server'}
/>

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
