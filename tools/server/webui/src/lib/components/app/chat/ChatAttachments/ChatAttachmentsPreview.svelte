<script lang="ts">
	import {
		ChevronLeft,
		ChevronRight,
		FileText,
		Image,
		Music,
		FileIcon,
		Eye,
		Info
	} from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { HorizontalScrollCarousel } from '$lib/components/app/misc';
	import { Button } from '$lib/components/ui/button';
	import * as Alert from '$lib/components/ui/alert';
	import type { DatabaseMessageExtra } from '$lib/types';
	import {
		getAttachmentDisplayItems,
		isImageFile,
		isAudioFile,
		isPdfFile,
		isTextFile,
		getLanguageFromFilename,
		createBase64DataUrl,
		formatFileSize
	} from '$lib/utils';
	import { convertPDFToImage } from '$lib/utils/browser-only';
	import { isMcpPrompt, isMcpResource } from '$lib/utils/attachment-display';
	import { modelsStore } from '$lib/stores/models.svelte';

	interface PreviewItem {
		id: string;
		name: string;
		size?: number;
		preview?: string;
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		textContent?: string;
		isImage: boolean;
		isAudio: boolean;
	}

	interface Props {
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		activeModelId?: string;
		class?: string;
		previewFocusIndex?: number;
	}

	let {
		uploadedFiles = [],
		attachments = [],
		activeModelId,
		class: className = '',
		previewFocusIndex = 0
	}: Props = $props();

	let allItems = $derived(
		getAttachmentDisplayItems({ uploadedFiles, attachments })
			.filter((item) => !isMcpPrompt(item) && !isMcpResource(item))
			.map(
				(item): PreviewItem => ({
					...item,
					isImage: isImageFile(item.attachment, item.uploadedFile),
					isAudio: isAudioFile(item.attachment, item.uploadedFile)
				})
			)
	);

	let currentIndex = $state(0);

	$effect(() => {
		if (previewFocusIndex >= 0 && previewFocusIndex < allItems.length) {
			currentIndex = previewFocusIndex;
		}
	});

	$effect(() => {
		const handler = (e: Event) => {
			const delta = (e as CustomEvent).detail;

			if (delta < 0) {
				currentIndex = currentIndex > 0 ? currentIndex - 1 : allItems.length - 1;
			} else {
				currentIndex = currentIndex < allItems.length - 1 ? currentIndex + 1 : 0;
			}
		};

		document.addEventListener('chat-attachments-nav', handler);

		return () => document.removeEventListener('chat-attachments-nav', handler);
	});

	$effect(() => {
		const index = currentIndex;
		setTimeout(() => {
			const thumbnail = document.querySelector(`[data-thumbnail-index="${index}"]`);

			thumbnail?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
		}, 0);
	});

	let currentItem = $derived(allItems[currentIndex] ?? null);
	let displayName = $derived(
		currentItem?.name ||
			currentItem?.uploadedFile?.name ||
			currentItem?.attachment?.name ||
			'Unknown File'
	);
	let isAudio = $derived(
		currentItem ? isAudioFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isImage = $derived(
		currentItem ? isImageFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isPdf = $derived(
		currentItem ? isPdfFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isText = $derived(
		currentItem ? isTextFile(currentItem.attachment, currentItem.uploadedFile) : false
	);

	let displayPreview = $derived(
		currentItem?.uploadedFile?.preview ||
			(isImage && currentItem?.attachment && 'base64Url' in currentItem.attachment
				? currentItem.attachment.base64Url
				: currentItem?.preview)
	);

	let displayTextContent = $derived(
		currentItem?.uploadedFile?.textContent ||
			(currentItem?.attachment && 'content' in currentItem.attachment
				? currentItem.attachment.content
				: currentItem?.textContent)
	);

	let language = $derived(getLanguageFromFilename(displayName));

	let fileSize = $derived(currentItem?.size ? formatFileSize(currentItem.size) : '');

	let hasVisionModality = $derived(
		currentItem && activeModelId ? modelsStore.modelSupportsVision(activeModelId) : false
	);

	let IconComponent = $derived(() => {
		if (isImage) return Image;
		if (isText || isPdf) return FileText;
		if (isAudio) return Music;
		return FileIcon;
	});

	let audioSrc = $derived(
		isAudio && currentItem
			? (currentItem.uploadedFile?.preview ??
					(currentItem.attachment &&
					'mimeType' in currentItem.attachment &&
					'base64Data' in currentItem.attachment
						? createBase64DataUrl(
								currentItem.attachment.mimeType,
								currentItem.attachment.base64Data
							)
						: null))
			: null
	);

	let pdfViewMode = $state<'text' | 'pages'>('pages');
	let pdfImages = $state<string[]>([]);
	let pdfImagesLoading = $state(false);
	let pdfImagesError = $state<string | null>(null);

	function resetPdfState() {
		pdfImages = [];
		pdfImagesLoading = false;
		pdfImagesError = null;
		pdfViewMode = 'pages';
	}

	$effect(() => {
		void currentIndex; // Needed to reset PDF state on every navigation, including PDF→PDF case

		resetPdfState();
	});

	function getFileExtension(name: string): string {
		const parts = name.split('.');
		if (parts.length > 1) {
			return parts.pop()?.toUpperCase() ?? '';
		}

		return '';
	}

	async function loadPdfImages() {
		if (!isPdf || pdfImages.length > 0 || pdfImagesLoading || !currentItem) return;

		pdfImagesLoading = true;
		pdfImagesError = null;

		try {
			let file: File | null = null;

			if (currentItem.uploadedFile?.file) {
				file = currentItem.uploadedFile.file;
			} else if (isPdf && currentItem.attachment) {
				// Check if we have pre-processed images
				if (
					'images' in currentItem.attachment &&
					currentItem.attachment.images &&
					Array.isArray(currentItem.attachment.images) &&
					currentItem.attachment.images.length > 0
				) {
					pdfImages = currentItem.attachment.images;
					return;
				}

				// Convert base64 back to File for processing
				if ('base64Data' in currentItem.attachment && currentItem.attachment.base64Data) {
					const base64Data = currentItem.attachment.base64Data;
					const byteCharacters = atob(base64Data);
					const byteNumbers = new Array(byteCharacters.length);
					for (let i = 0; i < byteCharacters.length; i++) {
						byteNumbers[i] = byteCharacters.charCodeAt(i);
					}
					const byteArray = new Uint8Array(byteNumbers);
					file = new File([byteArray], displayName, { type: 'application/pdf' });
				}
			}

			if (file) {
				pdfImages = await convertPDFToImage(file);
			} else {
				throw new Error('No PDF file available for conversion');
			}
		} catch (error) {
			pdfImagesError = error instanceof Error ? error.message : 'Failed to load PDF images';
		} finally {
			pdfImagesLoading = false;
		}
	}

	$effect(() => {
		// re-run on every navigation (handles PDF→PDF case)
		void currentIndex;
		if (isPdf && pdfViewMode === 'pages') {
			loadPdfImages();
		}
	});

	export function prev() {
		currentIndex = currentIndex > 0 ? currentIndex - 1 : allItems.length - 1;
	}

	export function next() {
		currentIndex = currentIndex < allItems.length - 1 ? currentIndex + 1 : 0;
	}

	function onNavigate(index: number) {
		currentIndex = index;
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="{className} flex flex-col text-white">
	<div class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
		{#if allItems.length > 1}
			<Button
				variant="secondary"
				size="icon"
				class="absolute top-1/2 left-4 z-10 h-8 w-8 -translate-y-1/2 rounded-full bg-background/5 p-0 text-white!"
				onclick={prev}
				aria-label="Previous"
			>
				<ChevronLeft class="size-4" />
			</Button>
		{/if}

		<div class="flex h-full w-full flex-col items-center justify-start overflow-auto py-4">
			{#if currentItem}
				<div
					class="sticky top-0 z-[20] mb-4 rounded-lg bg-black/5 px-4 py-2 text-center backdrop-blur-md"
				>
					<p class="font-medium text-white">{displayName}</p>

					<p class="text-xs text-white/60">{fileSize}</p>
				</div>

				{#if isPdf}
					<div class="mb-4 flex items-center justify-end gap-2">
						<Button
							variant={pdfViewMode === 'text' ? 'default' : 'outline'}
							size="sm"
							onclick={() => (pdfViewMode = 'text')}
							disabled={pdfImagesLoading}
						>
							<FileText class="mr-1 h-4 w-4" />

							Text
						</Button>

						<Button
							variant={pdfViewMode === 'pages' ? 'default' : 'outline'}
							size="sm"
							onclick={() => {
								pdfViewMode = 'pages';
								loadPdfImages();
							}}
							disabled={pdfImagesLoading}
						>
							{#if pdfImagesLoading}
								<div
									class="mr-1 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
								></div>
							{:else}
								<Eye class="mr-1 h-4 w-4" />
							{/if}

							Pages
						</Button>
					</div>
				{/if}

				{#if isImage && displayPreview}
					<div class="flex flex-1 items-center justify-center">
						<img
							src={displayPreview}
							alt={displayName}
							class="max-h-[80vh] max-w-[80vw] rounded-lg object-contain shadow-lg"
						/>
					</div>
				{:else if isPdf && pdfViewMode === 'pages'}
					{#if !hasVisionModality && activeModelId && currentItem}
						<Alert.Root class="mb-4">
							<Info class="h-4 w-4" />

							<Alert.Title>Preview only</Alert.Title>

							<Alert.Description>
								<span class="inline-flex">
									The selected model does not support vision. Only the extracted
									<!-- svelte-ignore a11y_click_events_have_key_events -->
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<span
										class="mx-1 cursor-pointer underline"
										onclick={() => (pdfViewMode = 'text')}
									>
										text
									</span>
									will be sent to the model.
								</span>
							</Alert.Description>
						</Alert.Root>
					{/if}

					{#if pdfImagesLoading}
						<div class="flex flex-1 items-center justify-center p-8">
							<div class="text-center">
								<div
									class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent"
								></div>

								<p class="text-white/70">Converting PDF to images...</p>
							</div>
						</div>
					{:else if pdfImagesError}
						<div class="flex flex-1 items-center justify-center p-8">
							<div class="text-center">
								<FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />

								<p class="mb-4 text-white/70">Failed to load PDF images</p>

								<p class="text-sm text-white/50">{pdfImagesError}</p>

								<Button class="mt-4" onclick={() => (pdfViewMode = 'text')}>View as Text</Button>
							</div>
						</div>
					{:else if pdfImages.length > 0}
						<!-- <div class="flex flex-1 flex-col overflow-auto px-4 pb-4"> -->
						{#each pdfImages as image, index (image)}
							<p class="mb-2 text-sm text-white/50">Page {index + 1}</p>

							<img
								src={image}
								alt="PDF Page {index + 1}"
								class="mx-auto max-w-[85vw] rounded-lg shadow-lg"
							/>

							<div class="h-4"></div>
						{/each}
						<!-- </div> -->
					{:else}
						<div class="flex flex-1 items-center justify-center p-8">
							<div class="text-center">
								<FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />

								<p class="text-white/70">No PDF pages available</p>
							</div>
						</div>
					{/if}
				{:else if (isText || (isPdf && pdfViewMode === 'text')) && displayTextContent}
					<div class="px-4 pb-4">
						<SyntaxHighlightedCode
							class="max-w-4xl"
							code={displayTextContent}
							{language}
							maxHeight="none"
						/>
					</div>
				{:else if isAudio}
					<div class="flex flex-1 items-center justify-center p-8">
						<div class="w-full max-w-md text-center">
							<Music class="mx-auto mb-4 h-16 w-16 text-white/50" />

							{#if audioSrc}
								<audio controls class="mb-4 w-full" src={audioSrc}>
									Your browser does not support the audio element.
								</audio>
							{:else}
								<p class="mb-4 text-white/70">Audio preview not available</p>
							{/if}

							<p class="text-sm text-white/50">{displayName}</p>
						</div>
					</div>
				{:else}
					<div class="flex flex-1 items-center justify-center p-8">
						<div class="text-center">
							{#if IconComponent}
								<IconComponent class="mx-auto mb-4 h-16 w-16 text-white/50" />
							{/if}

							<p class="text-white/70">Preview not available for this file type</p>
						</div>
					</div>
				{/if}
			{/if}

			{#if allItems.length > 1}
				<div class="sticky bottom-0 z-10 mt-4 flex-shrink-0">
					<HorizontalScrollCarousel class="max-w-full">
						{#each allItems as item, index (item.id)}
							<button
								data-thumbnail-index={index}
								class={[
									'relative flex-shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black/80 backdrop-blur-sm transition-all hover:opacity-90',
									index === currentIndex ? 'border-white' : 'border-transparent opacity-60',
									'[&:not(:first-child)]:last:mr-4 [&:not(:last-child)]:first:ml-4'
								]}
								onclick={() => onNavigate(index)}
								aria-label={`Go to ${item.name}`}
							>
								{#if item.isImage && item.preview}
									<img src={item.preview} alt={item.name} class="h-12 w-12 object-cover" />
								{:else}
									<div
										class="bg-foreground-muted/50 flex h-12 w-12 flex-col items-center justify-center gap-0.5 py-1"
									>
										{#if item.isAudio}
											<Music class="h-4 w-4 text-white/70" />
										{:else}
											<FileText class="h-4 w-4 text-white/70" />
										{/if}

										<span class="font-mono text-[9px] text-white/60"
											>{getFileExtension(item.name)}</span
										>
									</div>
								{/if}
							</button>
						{/each}
					</HorizontalScrollCarousel>
				</div>
			{/if}
		</div>

		{#if allItems.length > 1}
			<Button
				variant="secondary"
				size="icon"
				class="absolute top-1/2 right-4 z-10 h-8 w-8 -translate-y-1/2 rounded-full bg-background/5 p-0 text-white!"
				onclick={next}
				aria-label="Next"
			>
				<ChevronRight class="size-4" />
			</Button>
		{/if}
	</div>
</div>
