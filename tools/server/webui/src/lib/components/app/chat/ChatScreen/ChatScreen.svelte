<script lang="ts">
	import { parseFilesToMessageExtras, processFilesToChatUploaded } from '$lib/utils';
	import { serverStore } from '$lib/stores/server.svelte';
	import { ChatForm, ChatScreenHeader, ChatMessages, ServerInfo } from '$lib/components/app';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration
	} from '$lib/stores/chat.svelte';
	import { fade, fly, slide } from 'svelte/transition';
	import { AUTO_SCROLL_THRESHOLD } from '$lib/constants/auto-scroll';
	import { navigating } from '$app/state';
	import ChatScreenDragOverlay from './ChatScreenDragOverlay.svelte';

	let { showCenteredEmpty = false } = $props();
	let chatScrollContainer: HTMLDivElement | undefined = $state();
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let autoScrollEnabled = $state(true);
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let isDragOver = $state(false);
	let dragCounter = $state(0);

	const isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

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
		await sendMessage(message, extras);
		scrollChatToBottom();
		
		return true;
	}

	async function processFiles(files: File[]) {
		const processed = await processFilesToChatUploaded(files);
		uploadedFiles = [...uploadedFiles, ...processed];
	}

	function scrollChatToBottom(behavior: ScrollBehavior = 'smooth') {
		chatScrollContainer?.scrollTo({
			top: chatScrollContainer?.scrollHeight,
			behavior
		});
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
	})

	$effect(() => {
		if (isLoading() && autoScrollEnabled) {
			scrollInterval = setInterval(scrollChatToBottom, 200);
		} else if (scrollInterval) {
			clearInterval(scrollInterval);
			scrollInterval = undefined;
		}
	});
</script>

{#if isDragOver}
	<ChatScreenDragOverlay />
{/if}

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
					showHelperText={false}
					onSend={handleSendMessage}
					onStop={() => stopGeneration()}
					bind:uploadedFiles
					onFileUpload={handleFileUpload}
					onFileRemove={handleFileRemove}
				/>
			</div>
		</div>
	</div>
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
					{uploadedFiles}
				/>
			</div>
		</div>
	</div>
{/if}

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
