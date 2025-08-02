<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatForm, ChatMessages, ServerInfo } from '$lib/components';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration,
	} from '$lib/stores/chat.svelte';
	import { onMount } from 'svelte';
	import { fly, slide } from 'svelte/transition';
	import { Upload } from '@lucide/svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';
	import type { DatabaseMessageExtra } from '$lib/types/database.d.ts';

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

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]) {
		const extras = files ? await convertFilesToExtras(files) : undefined;
		await sendMessage(message, extras);
	}

	async function convertFilesToExtras(files: ChatUploadedFile[]): Promise<DatabaseMessageExtra[]> {
		const extras: DatabaseMessageExtra[] = [];
		
		for (const file of files) {
			if (file.type.startsWith('image/')) {
				if (file.preview) {
					extras.push({
						type: 'imageFile',
						name: file.name,
						base64Url: file.preview
					});
				}
			} else {
				// Handle text files and other non-image files
				try {
					const content = await readFileAsText(file.file);
					
					// Check if content is likely text (not binary)
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
		chatScrollContainer?.scrollTo({top: chatScrollContainer?.scrollHeight, behavior: 'instant'})
	}

	afterNavigate(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	})

	onMount(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	})


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
	})

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
				reader.onload = (e) => {
					uploadedFile.preview = e.target?.result as string;
					uploadedFiles = [...uploadedFiles, uploadedFile];
				};
				reader.readAsDataURL(file);
			} else {
				uploadedFiles = [...uploadedFiles, uploadedFile];
			}
		}
	}

	function handleFileUpload(files: File[]) {
		processFiles(files);
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);
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

	/**
	 * Read a file as text content
	 */
	async function readFileAsText(file: File): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = (event) => {
				if (event.target?.result) {
					resolve(event.target.result as string);
				} else {
					reject(new Error('Failed to read file'));
				}
			};
			reader.onerror = () => reject(new Error('File reading error'));
			reader.readAsText(file);
		});
	}

	/**
	 * Simple heuristic to determine if content is likely text (not binary)
	 * Based on webui-old's isLikelyNotBinary function but simplified
	 */
	function isLikelyTextFile(content: string): boolean {
		if (!content) return true;
		
		// Check first 1000 characters for binary indicators
		const sample = content.substring(0, 1000);
		let suspiciousCount = 0;
		let nullCount = 0;
		
		for (let i = 0; i < sample.length; i++) {
			const charCode = sample.charCodeAt(i);
			
			// Count null bytes
			if (charCode === 0) {
				nullCount++;
				suspiciousCount++;
				continue;
			}
			
			// Count suspicious control characters (excluding common ones like tab, newline, carriage return)
			if (charCode < 32 && charCode !== 9 && charCode !== 10 && charCode !== 13) {
				suspiciousCount++;
			}
			
			// Count replacement characters (indicates encoding issues)
			if (charCode === 0xFFFD) {
				suspiciousCount++;
			}
		}
		
		// Reject if too many null bytes or suspicious characters
		if (nullCount > 2) return false;
		if (suspiciousCount / sample.length > 0.1) return false;
		
		return true;
	}
</script>

{#if isDragOver}
	<div class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
		<div class="bg-background border-border flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 shadow-lg">
			<Upload class="text-muted-foreground mb-4 h-12 w-12" />
			<p class="text-foreground text-lg font-medium">Attach a file</p>
			<p class="text-muted-foreground text-sm">Drop your files here to upload</p>
		</div>
	</div>
{/if}

{#if !isEmpty}
	<div 
		class="flex h-full flex-col overflow-y-auto" 
		bind:this={chatScrollContainer} 
		onscroll={handleScroll}
		ondragenter={handleDragEnter}
		ondragleave={handleDragLeave}
		ondragover={handleDragOver}
		ondrop={handleDrop}
		role="main"
		aria-label="Chat interface with file drop zone"
	>
			<ChatMessages class="mb-36" messages={activeMessages()} />

			<div
				class="z-999 sticky bottom-0 mx-auto mt-auto max-w-[56rem]"
				in:slide={{ duration: 400, axis: 'y' }}
			>
				<div class="bg-background m-auto rounded-t-3xl border-t pb-4 min-w-[56rem]">
					<ChatForm
						isLoading={isLoading()}
						showHelperText={false}
						onSend={handleSendMessage}
						onStop={() => stopGeneration()}
						bind:uploadedFiles={uploadedFiles}
						onFileUpload={handleFileUpload}
						onFileRemove={handleFileRemove}
					/>
				</div>
			</div>
	</div>
{:else}
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
			<div class="mb-8 text-center" in:fly={{ y: -30, duration: 600 }}>
				<h1 class="mb-2 text-3xl font-semibold tracking-tight">llama.cpp</h1>

				<p class="text-muted-foreground text-lg">How can I help you today?</p>
			</div>

			<div
				class="mb-6 flex justify-center"
				in:slide={{ duration: 500, delay: 300, axis: 'y' }}
			>
				<ServerInfo />
			</div>

			<div in:slide={{ duration: 600, delay: 500, axis: 'y' }}>
				<ChatForm
					isLoading={isLoading()}
					showHelperText={true}
					onSend={handleSendMessage}
					onStop={() => stopGeneration()}
					uploadedFiles={uploadedFiles}
					onFileUpload={handleFileUpload}
					onFileRemove={handleFileRemove}
				/>
			</div>
		</div>
	</div>
{/if}
