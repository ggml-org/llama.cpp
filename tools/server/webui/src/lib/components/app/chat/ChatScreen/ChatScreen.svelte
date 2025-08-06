<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatForm, ChatHeader, ChatMessages, ServerInfo } from '$lib/components/app';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration
	} from '$lib/stores/chat.svelte';
	import { onMount } from 'svelte';
	import { fade, fly, slide } from 'svelte/transition';
	import { Upload } from '@lucide/svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';
	import type { DatabaseMessageExtra } from '$lib/types/database.d.ts';
	import {
		convertPDFToText,
		isLikelyTextFile,
		isPdfMimeType,
		isSvgMimeType,
		isTextFileByName,
		readFileAsText,
		svgBase64UrlToPngDataURL
	} from '$lib/utils';
	import { serverStore } from '$lib/stores/server.svelte';

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

	function validateChatUploadedFiles(files?: ChatUploadedFile[]): boolean {
		if (!files) return true;

		for (const file of files) {
			if (file.type === 'image/webp' || file.name.toLowerCase().endsWith('.webp')) {
				alert(
					'WebP format is not supported at the moment. Please use JPEG or PNG images instead.'
				);
				return false;
			}
		}

		return true;
	}

	async function handleSendMessage(
		message: string,
		files?: ChatUploadedFile[]
	): Promise<boolean> {
		if (!validateChatUploadedFiles(files)) {
			return false;
		}

		const extras = files ? await convertFilesToExtras(files) : undefined;
		await sendMessage(message, extras);
		return true;
	}

	async function convertFilesToExtras(
		files: ChatUploadedFile[]
	): Promise<DatabaseMessageExtra[]> {
		const extras: DatabaseMessageExtra[] = [];

		for (const file of files) {
			if (file.type.startsWith('image/')) {
				if (file.preview) {
					let base64Url = file.preview;

					if (isSvgMimeType(file.type)) {
						try {
							base64Url = await svgBase64UrlToPngDataURL(base64Url);
						} catch (error) {
							console.error(
								'Failed to convert SVG to PNG for database storage:',
								error
							);
						}
					}

					extras.push({
						type: 'imageFile',
						name: file.name,
						base64Url
					});
				}
			} else if (isPdfMimeType(file.type)) {
				try {
					// For now, always process PDF as text
					// todo: Add settings to allow PDF as images for vision models
					const content = await convertPDFToText(file.file);

					extras.push({
						type: 'pdfFile',
						name: file.name,
						content: content,
						processedAsImages: false
					});
				} catch (error) {
					console.error(`Failed to process PDF file ${file.name}:`, error);
				}
			} else {
				try {
					const content = await readFileAsText(file.file);

					if (isLikelyTextFile(content)) {
						extras.push({
							type: 'textFile',
							name: file.name,
							content: content
						});
					} else {
						console.warn(`File ${file.name} appears to be binary and will be skipped`);
					}
				} catch (error) {
					console.error(`Failed to read file ${file.name}:`, error);
				}
			}
		}

		return extras;
	}

	function scrollChatToBottom() {
		chatScrollContainer?.scrollTo({
			top: chatScrollContainer?.scrollHeight,
			behavior: 'instant'
		});
	}

	afterNavigate(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	});

	onMount(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	});

	function handleScroll() {
		if (!chatScrollContainer) return;

		const { scrollTop, scrollHeight, clientHeight } = chatScrollContainer;
		const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

		if (distanceFromBottom > 50) {
			autoScrollEnabled = false;
		} else if (distanceFromBottom <= 1) {
			autoScrollEnabled = true;
		}
	}

	$effect(() => {
		if (isLoading() && autoScrollEnabled) {
			scrollInterval = setInterval(scrollChatToBottom, 100);
		} else if (scrollInterval) {
			clearInterval(scrollInterval);
			scrollInterval = undefined;
		}
	});

	function processFiles(files: File[]) {
		for (const file of files) {
			const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
			const uploadedFile: ChatUploadedFile = {
				id,
				name: file.name,
				size: file.size,
				type: file.type,
				file
			};

			if (file.type.startsWith('image/')) {
				const reader = new FileReader();
				reader.onload = async (e) => {
					let preview = e.target?.result as string;

					// Convert SVG to PNG if necessary
					if (isSvgMimeType(file.type)) {
						try {
							preview = await svgBase64UrlToPngDataURL(preview);
						} catch (error) {
							console.error('Failed to convert SVG to PNG:', error);
							// Use original SVG preview if conversion fails
						}
					}

					uploadedFile.preview = preview;
					uploadedFiles = [...uploadedFiles, uploadedFile];
				};
				reader.readAsDataURL(file);
			} else if (file.type.startsWith('text/') || isTextFileByName(file.name)) {
				const reader = new FileReader();
				reader.onload = (e) => {
					const content = e.target?.result as string;
					if (content) {
						uploadedFile.textContent = content;
					}
					uploadedFiles = [...uploadedFiles, uploadedFile];
				};
				reader.onerror = () => {
					// If reading fails, still add the file without text content
					uploadedFiles = [...uploadedFiles, uploadedFile];
				};
				reader.readAsText(file);
			} else if (isPdfMimeType(file.type)) {
				// For PDF files, we'll process them during conversion to extras
				// Just add them to the list for now
				uploadedFiles = [...uploadedFiles, uploadedFile];
			} else {
				uploadedFiles = [...uploadedFiles, uploadedFile];
			}
		}
	}

	function handleFileUpload(files: File[]) {
		processFiles(files);
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
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
</script>

{#if isDragOver}
	<div
		class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
	>
		<div
			class="bg-background border-border flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 shadow-lg"
		>
			<Upload class="text-muted-foreground mb-4 h-12 w-12" />
			<p class="text-foreground text-lg font-medium">Attach a file</p>
			<p class="text-muted-foreground text-sm">Drop your files here to upload</p>
		</div>
	</div>
{/if}

<ChatHeader />

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
		class="flex h-full items-center justify-center"
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		role="main"
		aria-label="Welcome screen with file drop zone"
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
					showHelperText={true}
					onSend={handleSendMessage}
					onStop={() => stopGeneration()}
					{uploadedFiles}
					onFileUpload={handleFileUpload}
					onFileRemove={handleFileRemove}
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
