<script lang="ts">
	import { FileText, Image, Music, FileIcon, Eye, Info } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as Alert from '$lib/components/ui/alert';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import type { DatabaseMessageExtra } from '$lib/types';
	import {
		getAttachmentDisplayItems,
		isImageFile,
		isAudioFile,
		isPdfFile,
		isTextFile,
		getLanguageFromFilename,
		createBase64DataUrl
	} from '$lib/utils';
	import { convertPDFToImage } from '$lib/utils/browser-only';
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
			.filter((item) => !item.isMcpPrompt && !item.isMcpResource)
			.map(
				(item): PreviewItem => ({
					...item,
					isImage: isImageFile(item.attachment, item.uploadedFile)
				})
			)
	);

	let isGallery = $derived(allItems.length > 1);
	let currentIndex = $state(0);

	const thumbnailCarousel = useScrollCarousel();

	$effect(() => {
		if (previewFocusIndex >= 0 && previewFocusIndex < allItems.length) {
			currentIndex = previewFocusIndex;

			// Scroll the focused thumbnail into view after a tick
			setTimeout(() => {
				const thumbnail = document.querySelector(`[data-thumbnail-index="${currentIndex}"]`);
				if (thumbnail && thumbnailCarousel.scrollContainer) {
					thumbnailCarousel.scrollToCenter(thumbnail as HTMLElement);
				}
			}, 0);
		}
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
		resetPdfState();
	});

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
		if (isPdf && pdfViewMode === 'pages') {
			loadPdfImages();
		}
	});

	function prevItem() {
		if (currentIndex > 0) currentIndex--;
	}

	function nextItem() {
		if (currentIndex < allItems.length - 1) currentIndex++;
	}
</script>

<div class="{className} flex flex-col">
	{#if isGallery}
		<!-- Gallery mode with carousel -->
		<div class="relative flex items-center justify-center overflow-hidden">
			<!-- Left arrow -->
			<button
				class="absolute top-1/2 left-2 z-10 -translate-y-1/2 rounded-full bg-background/80 p-2 backdrop-blur-sm transition-opacity hover:bg-background disabled:pointer-events-none disabled:opacity-0"
				disabled={currentIndex <= 0}
				onclick={prevItem}
				aria-label="Previous"
			>
				<span class="text-lg leading-none">‹</span>
			</button>

			<!-- Preview container -->
			<div class="flex w-full items-center justify-center px-12">
				{#if currentItem}
					<div class="space-y-4">
						<!-- PDF view mode toggle -->
						{#if isPdf}
							<div class="flex items-center justify-end gap-6">
								<div class="flex items-center gap-2">
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
							</div>
						{/if}

						<!-- Preview content -->
						<div class="flex-1 overflow-auto">
							{#if isImage && displayPreview}
								<div class="flex items-center justify-center">
									<img
										src={displayPreview}
										alt={displayName}
										class="max-h-full rounded-lg object-contain shadow-lg"
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
									<div class="flex items-center justify-center p-8">
										<div class="text-center">
											<div
												class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"
											></div>
											<p class="text-muted-foreground">Converting PDF to images...</p>
										</div>
									</div>
								{:else if pdfImagesError}
									<div class="flex items-center justify-center p-8">
										<div class="text-center">
											<FileText class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
											<p class="mb-4 text-muted-foreground">Failed to load PDF images</p>
											<p class="text-sm text-muted-foreground">{pdfImagesError}</p>
											<Button class="mt-4" onclick={() => (pdfViewMode = 'text')}
												>View as Text</Button
											>
										</div>
									</div>
								{:else if pdfImages.length > 0}
									<div class="max-h-[70vh] space-y-4 overflow-auto">
										{#each pdfImages as image, index (image)}
											<div class="text-center">
												<p class="mb-2 text-sm text-muted-foreground">Page {index + 1}</p>
												<img
													src={image}
													alt="PDF Page {index + 1}"
													class="mx-auto max-w-full rounded-lg shadow-lg"
												/>
											</div>
										{/each}
									</div>
								{:else}
									<div class="flex items-center justify-center p-8">
										<div class="text-center">
											<FileText class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
											<p class="mb-4 text-muted-foreground">No PDF pages available</p>
										</div>
									</div>
								{/if}
							{:else if (isText || (isPdf && pdfViewMode === 'text')) && displayTextContent}
								<SyntaxHighlightedCode
									code={displayTextContent}
									{language}
									maxWidth="calc(69rem - 2rem)"
								/>
							{:else if isAudio}
								<div class="flex items-center justify-center p-8">
									<div class="w-full max-w-md text-center">
										<Music class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
										{#if audioSrc}
											<audio controls class="mb-4 w-full" src={audioSrc}>
												Your browser does not support the audio element.
											</audio>
										{:else}
											<p class="mb-4 text-muted-foreground">Audio preview not available</p>
										{/if}
										<p class="text-sm text-muted-foreground">{displayName}</p>
									</div>
								</div>
							{:else}
								<div class="flex items-center justify-center p-8">
									<div class="text-center">
										{#if IconComponent}
											<IconComponent class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
										{/if}
										<p class="mb-4 text-muted-foreground">
											Preview not available for this file type
										</p>
									</div>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			</div>

			<!-- Right arrow -->
			<button
				class="absolute top-1/2 right-2 z-10 -translate-y-1/2 rounded-full bg-background/80 p-2 backdrop-blur-sm transition-opacity hover:bg-background disabled:pointer-events-none disabled:opacity-0"
				disabled={currentIndex >= allItems.length - 1}
				onclick={nextItem}
				aria-label="Next"
			>
				<span class="text-lg leading-none">›</span>
			</button>
		</div>

		<!-- Thumbnails strip -->
		<div
			bind:this={thumbnailCarousel.scrollContainer}
			onscroll={thumbnailCarousel.updateScrollButtons}
			class="flex gap-2 overflow-x-auto px-2"
		>
			{#each allItems as item, index (item.id)}
				<button
					data-thumbnail-index={index}
					class="relative flex-shrink-0 overflow-hidden rounded border-2 transition-colors hover:opacity-80 {index ===
					currentIndex
						? 'border-primary'
						: 'border-transparent opacity-60'}"
					onclick={() => {
						currentIndex = index;
						setTimeout(() => {
							const thumbnail = document.querySelector(`[data-thumbnail-index="${index}"]`);
							if (thumbnail && thumbnailCarousel.scrollContainer) {
								thumbnailCarousel.scrollToCenter(thumbnail as HTMLElement);
							}
						}, 0);
					}}
					aria-label={`Go to ${item.name}`}
				>
					{#if item.isImage && item.preview}
						<img src={item.preview} alt={item.name} class="h-12 w-12 object-cover" />
					{:else}
						<div class="flex h-12 w-12 items-center justify-center bg-muted">
							<span class="text-xs text-muted-foreground">File</span>
						</div>
					{/if}
				</button>
			{/each}
		</div>
	{:else}
		<!-- Single item mode - no carousel, just the preview -->
		{#if allItems.length === 1 && currentItem}
			<div class="space-y-4">
				<!-- PDF view mode toggle -->
				{#if isPdf}
					<div class="flex items-center justify-end gap-6">
						<div class="flex items-center gap-2">
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
					</div>
				{/if}

				<!-- Preview content -->
				<div class="flex-1 overflow-auto">
					{#if isImage && displayPreview}
						<div class="flex items-center justify-center">
							<img
								src={displayPreview}
								alt={displayName}
								class="max-h-full rounded-lg object-contain shadow-lg"
							/>
						</div>
					{:else if isPdf && pdfViewMode === 'pages'}
						{#if !hasVisionModality && activeModelId}
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
							<div class="flex items-center justify-center p-8">
								<div class="text-center">
									<div
										class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"
									></div>
									<p class="text-muted-foreground">Converting PDF to images...</p>
								</div>
							</div>
						{:else if pdfImagesError}
							<div class="flex items-center justify-center p-8">
								<div class="text-center">
									<FileText class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
									<p class="mb-4 text-muted-foreground">Failed to load PDF images</p>
									<p class="text-sm text-muted-foreground">{pdfImagesError}</p>
									<Button class="mt-4" onclick={() => (pdfViewMode = 'text')}>View as Text</Button>
								</div>
							</div>
						{:else if pdfImages.length > 0}
							<div class="max-h-[70vh] space-y-4 overflow-auto">
								{#each pdfImages as image, index (image)}
									<div class="text-center">
										<p class="mb-2 text-sm text-muted-foreground">Page {index + 1}</p>
										<img
											src={image}
											alt="PDF Page {index + 1}"
											class="mx-auto max-w-full rounded-lg shadow-lg"
										/>
									</div>
								{/each}
							</div>
						{:else}
							<div class="flex items-center justify-center p-8">
								<div class="text-center">
									<FileText class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
									<p class="mb-4 text-muted-foreground">No PDF pages available</p>
								</div>
							</div>
						{/if}
					{:else if (isText || (isPdf && pdfViewMode === 'text')) && displayTextContent}
						<SyntaxHighlightedCode
							code={displayTextContent}
							{language}
							maxWidth="calc(69rem - 2rem)"
						/>
					{:else if isAudio}
						<div class="flex items-center justify-center p-8">
							<div class="w-full max-w-md text-center">
								<Music class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
								{#if audioSrc}
									<audio controls class="mb-4 w-full" src={audioSrc}>
										Your browser does not support the audio element.
									</audio>
								{:else}
									<p class="mb-4 text-muted-foreground">Audio preview not available</p>
								{/if}
								<p class="text-sm text-muted-foreground">{displayName}</p>
							</div>
						</div>
					{:else}
						<div class="flex items-center justify-center p-8">
							<div class="text-center">
								{#if IconComponent}
									<IconComponent class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />
								{/if}
								<p class="mb-4 text-muted-foreground">Preview not available for this file type</p>
							</div>
						</div>
					{/if}
				</div>
			</div>
		{/if}
	{/if}
</div>
