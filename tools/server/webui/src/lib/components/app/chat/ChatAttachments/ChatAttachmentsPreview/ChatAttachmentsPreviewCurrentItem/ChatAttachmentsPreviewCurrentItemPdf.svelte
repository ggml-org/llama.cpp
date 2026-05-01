<script lang="ts">
	import type { ChatAttachmentDisplayItem } from '$lib/types';
	import { FileText, Eye, Info } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Alert from '$lib/components/ui/alert';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { getLanguageFromFilename } from '$lib/utils';
	import { convertPDFToImage } from '$lib/utils/browser-only';
	import { PdfViewMode } from '$lib/enums';

	interface Props {
		currentItem: ChatAttachmentDisplayItem | null;
		displayName: string;
		displayTextContent: string | undefined;
		hasVisionModality: boolean;
		activeModelId?: string;
	}

	let { currentItem, displayName, displayTextContent, hasVisionModality, activeModelId }: Props =
		$props();

	let pdfViewMode = $state<PdfViewMode>(PdfViewMode.PAGES);
	let pdfImages = $state<string[]>([]);
	let pdfImagesLoading = $state(false);
	let pdfImagesError = $state<string | null>(null);

	let language = $derived(getLanguageFromFilename(displayName));

	async function loadPdfImages() {
		if (pdfImages.length > 0 || pdfImagesLoading || !currentItem) return;

		pdfImagesLoading = true;
		pdfImagesError = null;

		try {
			let file: File | null = null;

			if (currentItem.uploadedFile?.file) {
				file = currentItem.uploadedFile.file;
			} else if (currentItem.attachment) {
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
		if (pdfViewMode === PdfViewMode.PAGES) {
			loadPdfImages();
		}
	});
</script>

<div class="mb-4 flex items-center justify-end gap-2">
	<Button
		variant={pdfViewMode === PdfViewMode.TEXT ? 'default' : 'outline'}
		size="sm"
		onclick={() => (pdfViewMode = PdfViewMode.TEXT)}
		disabled={pdfImagesLoading}
	>
		<FileText class="mr-1 h-4 w-4" />
		Text
	</Button>

	<Button
		variant={pdfViewMode === PdfViewMode.PAGES ? 'default' : 'outline'}
		size="sm"
		onclick={() => (pdfViewMode = PdfViewMode.PAGES)}
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

{#if !hasVisionModality && activeModelId && currentItem}
	<Alert.Root class="mb-4 max-w-4xl">
		<Info class="h-4 w-4" />
		<Alert.Title>Preview only</Alert.Title>
		<Alert.Description>
			<span class="inline-flex">
				The selected model does not support vision. Only the extracted
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<span
					class="mx-1 cursor-pointer underline"
					onclick={() => (pdfViewMode = PdfViewMode.TEXT)}
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
		</div>
	</div>
{:else if pdfImages.length > 0}
	{#each pdfImages as image, index (image)}
		<p class="mb-2 text-sm text-white/50">Page {index + 1}</p>
		<img src={image} alt="PDF Page {index + 1}" class="mx-auto max-w-[85vw] rounded-lg shadow-lg" />
		<div class="h-4"></div>
	{/each}
{:else}
	<div class="flex flex-1 items-center justify-center p-8">
		<div class="text-center">
			<FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />
			<p class="text-white/70">No PDF pages available</p>
		</div>
	</div>
{/if}

{#if pdfViewMode === PdfViewMode.TEXT && displayTextContent}
	<div class="px-4 pb-4">
		<SyntaxHighlightedCode
			class="max-w-4xl"
			code={displayTextContent}
			{language}
			maxHeight="none"
		/>
	</div>
{/if}
