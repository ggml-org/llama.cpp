<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { FileText, Image, Music, FileIcon, Eye } from '@lucide/svelte';
	import { convertPDFToImage } from '$lib/utils/pdf-processing';
	import { Button } from '$lib/components/ui/button';
	import { FileTypeCategory, PdfMimeType, getFileTypeCategory } from '$lib/constants/supported-file-types';

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

	let displayType = $derived(
		uploadedFile?.type ||
			(attachment?.type === 'imageFile'
				? 'image'
				: attachment?.type === 'textFile'
					? 'text'
					: attachment?.type === 'audioFile'
						? (attachment as any).mimeType || 'audio'
						: attachment?.type === 'pdfFile'
							? 'application/pdf'
							: type || 'unknown')
	);

	let displaySize = $derived(uploadedFile?.size || size);

	let displayPreview = $derived(
		uploadedFile?.preview ||
			(attachment?.type === 'imageFile' ? (attachment as any).base64Url : preview)
	);

	let displayTextContent = $derived(
		uploadedFile?.textContent ||
			(attachment?.type === 'textFile'
				? (attachment as any).content
				: attachment?.type === 'pdfFile'
					? (attachment as any).content
					: textContent)
	);

	let isImage = $derived(getFileTypeCategory(displayType) === FileTypeCategory.IMAGE || displayType === 'image');
	let isText = $derived(getFileTypeCategory(displayType) === FileTypeCategory.TEXT || displayType === 'text');
	let isPdf = $derived(displayType === PdfMimeType.PDF);
	let isAudio = $derived(getFileTypeCategory(displayType) === FileTypeCategory.AUDIO || displayType === 'audio');

	let pdfViewMode = $state<'text' | 'pages'>('pages');
	let pdfImages = $state<string[]>([]);
	let pdfImagesLoading = $state(false);
	let pdfImagesError = $state<string | null>(null);

	let IconComponent = $derived(() => {
		if (isImage) return Image;
		if (isText || isPdf) return FileText;
		if (isAudio) return Music;
		return FileIcon;
	});

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';

		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));

		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

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
				if ((attachment as any).images && Array.isArray((attachment as any).images)) {
					pdfImages = (attachment as any).images;
					return;
				}
				
				// Convert base64 back to File for processing
				if ((attachment as any).base64Data) {
					const base64Data = (attachment as any).base64Data;
					const byteCharacters = atob(base64Data);
					const byteNumbers = new Array(byteCharacters.length);
					for (let i = 0; i < byteCharacters.length; i++) {
						byteNumbers[i] = byteCharacters.charCodeAt(i);
					}
					const byteArray = new Uint8Array(byteNumbers);
					file = new File([byteArray], displayName, { type: PdfMimeType.PDF });
				}
			}
			
			if (file) {
				const images = await convertPDFToImage(file);
				pdfImages = images;
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
	<Dialog.Content class="grid max-h-[90vh] !p-10 max-w-5xl overflow-hidden sm:w-auto sm:max-w-6xl">
		<Dialog.Header class="flex-shrink-0">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-3">
					{#if IconComponent}
						<IconComponent class="text-muted-foreground h-5 w-5" />
					{/if}

					<div>
						<Dialog.Title class="text-left">{displayName}</Dialog.Title>

						<div class="text-muted-foreground flex items-center gap-2 text-sm">
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
							onclick={() => pdfViewMode = 'text'}
							disabled={pdfImagesLoading}
						>
							<FileText class="h-4 w-4 mr-1" />
							Text
						</Button>
						<Button
							variant={pdfViewMode === 'pages' ? 'default' : 'outline'}
							size="sm"
							onclick={() => { pdfViewMode = 'pages'; loadPdfImages(); }}
							disabled={pdfImagesLoading}
						>
							{#if pdfImagesLoading}
								<div class="h-4 w-4 mr-1 animate-spin rounded-full border-2 border-current border-t-transparent"></div>
							{:else}
								<Eye class="h-4 w-4 mr-1" />
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
							<div class="h-8 w-8 mx-auto mb-4 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
							<p class="text-muted-foreground">Converting PDF to images...</p>
						</div>
					</div>
				{:else if pdfImagesError}
					<div class="flex items-center justify-center p-8">
						<div class="text-center">
							<FileText class="text-muted-foreground mx-auto mb-4 h-16 w-16" />
							<p class="text-muted-foreground mb-4">Failed to load PDF images</p>
							<p class="text-muted-foreground text-sm">{pdfImagesError}</p>
							<Button class="mt-4" onclick={() => { pdfViewMode = 'text'; }}>View as Text</Button>
						</div>
					</div>
				{:else if pdfImages.length > 0}
					<div class="max-h-[70vh] overflow-auto space-y-4">
						{#each pdfImages as image, index}
							<div class="text-center">
								<p class="text-muted-foreground mb-2 text-sm">Page {index + 1}</p>
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
							<FileText class="text-muted-foreground mx-auto mb-4 h-16 w-16" />
							<p class="text-muted-foreground mb-4">No PDF pages available</p>
						</div>
					</div>
				{/if}
			{:else if (isText || (isPdf && pdfViewMode === 'text')) && displayTextContent}
				<div
					class="bg-muted max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-lg p-4 font-mono text-sm"
				>
					{displayTextContent}
				</div>
			{:else if isAudio}
				<div class="flex items-center justify-center p-8">
					<div class="w-full max-w-md text-center">
						<Music class="text-muted-foreground mx-auto mb-4 h-16 w-16" />
						
						{#if attachment?.type === 'audioFile'}
							<audio 
								controls 
								class="w-full mb-4"
								src="data:{(attachment as any).mimeType};base64,{(attachment as any).base64Data}"
							>
								Your browser does not support the audio element.
							</audio>
						{:else if uploadedFile?.preview}
							<audio 
								controls 
								class="w-full mb-4"
								src={uploadedFile.preview}
							>
								Your browser does not support the audio element.
							</audio>
						{:else}
							<p class="text-muted-foreground mb-4">Audio preview not available</p>
						{/if}
						
						<p class="text-muted-foreground text-sm">
							{displayName}
						</p>
					</div>
				</div>
			{:else}
				<div class="flex items-center justify-center p-8">
					<div class="text-center">
						{#if IconComponent}
							<IconComponent class="text-muted-foreground mx-auto mb-4 h-16 w-16" />
						{/if}

						<p class="text-muted-foreground mb-4">
							Preview not available for this file type
						</p>
					</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
