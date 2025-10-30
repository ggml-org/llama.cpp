<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { FileText, Image, Music, FileIcon, Eye } from '@lucide/svelte';
	import { FileTypeCategory, MimeTypeApplication } from '$lib/enums/files';
	import { convertPDFToImage } from '$lib/utils/pdf-processing';
	import { Button } from '$lib/components/ui/button';
	import { getFileTypeCategory } from '$lib/utils/file-type';
	import { formatFileSize } from '$lib/utils/file-preview';

	interface Props {
		open: boolean;
		// Either an uploaded file or a stored attachment
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		// For uploaded files
		preview?: string;
		name?: string;
		type?: string;
		size?: number;
		textContent?: string;
	}

	let {
		open = $bindable(),
		uploadedFile,
		attachment,
		preview,
		name,
		type,
		size,
		textContent
	}: Props = $props();

	let displayName = $derived(uploadedFile?.name || attachment?.name || name || 'Unknown File');

	let displayPreview = $derived(
		uploadedFile?.preview || (attachment?.type === 'imageFile' ? attachment.base64Url : preview)
	);

	let displayType = $derived(
		uploadedFile?.type ||
			(attachment?.type === 'imageFile'
				? 'image'
				: attachment?.type === 'textFile'
					? 'text'
					: attachment?.type === 'audioFile'
						? attachment.mimeType || 'audio'
						: attachment?.type === 'pdfFile'
							? MimeTypeApplication.PDF
							: type || 'unknown')
	);

	let displaySize = $derived(uploadedFile?.size || size);

	let displayTextContent = $derived(
		uploadedFile?.textContent ||
			(attachment?.type === 'textFile'
				? attachment.content
				: attachment?.type === 'pdfFile'
					? attachment.content
					: textContent)
	);

	let isAudio = $derived(
		getFileTypeCategory(displayType) === FileTypeCategory.AUDIO || displayType === 'audio'
	);

	let isImage = $derived(
		getFileTypeCategory(displayType) === FileTypeCategory.IMAGE || displayType === 'image'
	);

	let isPdf = $derived(displayType === MimeTypeApplication.PDF);

	let isText = $derived(
		getFileTypeCategory(displayType) === FileTypeCategory.TEXT || displayType === 'text'
	);

	let IconComponent = $derived(() => {
		if (isImage) return Image;
		if (isText || isPdf) return FileText;
		if (isAudio) return Music;

		return FileIcon;
	});

	let pdfViewMode = $state<'text' | 'pages'>('pages');

	let pdfImages = $state<string[]>([]);

	let pdfImagesLoading = $state(false);

	let pdfImagesError = $state<string | null>(null);

	async function loadPdfImages() {
		if (!isPdf || pdfImages.length > 0 || pdfImagesLoading) return;

		pdfImagesLoading = true;
		pdfImagesError = null;

		try {
			let file: File | null = null;

			if (uploadedFile?.file) {
				file = uploadedFile.file;
			} else if (attachment?.type === 'pdfFile') {
				// Check if we have pre-processed images
				if (attachment.images && Array.isArray(attachment.images)) {
					pdfImages = attachment.images;
					return;
				}

				// Convert base64 back to File for processing
				if (attachment.base64Data) {
					const base64Data = attachment.base64Data;
					const byteCharacters = atob(base64Data);
					const byteNumbers = new Array(byteCharacters.length);
					for (let i = 0; i < byteCharacters.length; i++) {
						byteNumbers[i] = byteCharacters.charCodeAt(i);
					}
					const byteArray = new Uint8Array(byteNumbers);
					file = new File([byteArray], displayName, { type: MimeTypeApplication.PDF });
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
		if (open && isPdf && pdfViewMode === 'pages') {
			loadPdfImages();
		}
	});
</script>

<Dialog.Root bind:open>
	<Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden !p-10 sm:w-auto sm:max-w-6xl">
		<Dialog.Header class="flex-shrink-0">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-3">
					{#if IconComponent}
						<IconComponent class="h-5 w-5 text-muted-foreground" />
					{/if}

					<div>
						<Dialog.Title class="text-left">{displayName}</Dialog.Title>

						<div class="flex items-center gap-2 text-sm text-muted-foreground">
							<span>{displayType}</span>

							{#if displaySize}
								<span>•</span>

								<span>{formatFileSize(displaySize)}</span>
							{/if}
						</div>
					</div>
				</div>

				{#if isPdf}
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
				{/if}
			</div>
		</Dialog.Header>

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
				<div
					class="max-h-[60vh] overflow-auto rounded-lg bg-muted p-4 font-mono text-sm break-words whitespace-pre-wrap"
				>
					{displayTextContent}
				</div>
			{:else if isAudio}
				<div class="flex items-center justify-center p-8">
					<div class="w-full max-w-md text-center">
						<Music class="mx-auto mb-4 h-16 w-16 text-muted-foreground" />

						{#if attachment?.type === 'audioFile'}
							<audio
								controls
								class="mb-4 w-full"
								src="data:{attachment.mimeType};base64,{attachment.base64Data}"
							>
								Your browser does not support the audio element.
							</audio>
						{:else if uploadedFile?.preview}
							<audio controls class="mb-4 w-full" src={uploadedFile.preview}>
								Your browser does not support the audio element.
							</audio>
						{:else}
							<p class="mb-4 text-muted-foreground">Audio preview not available</p>
						{/if}

						<p class="text-sm text-muted-foreground">
							{displayName}
						</p>
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
	</Dialog.Content>
</Dialog.Root>
