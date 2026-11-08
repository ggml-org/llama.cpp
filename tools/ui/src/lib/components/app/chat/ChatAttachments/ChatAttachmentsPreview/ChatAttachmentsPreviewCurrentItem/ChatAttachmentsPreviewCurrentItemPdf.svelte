<script lang="ts">
	import { Eye, FileText, Info } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import * as Alert from '$lib/components/ui/alert';
	import { Button } from '$lib/components/ui/button';
	import { ICON_CLASS_DEFAULT, PDF_AS_IMAGES_LEGACY_CONTENT_REGEX } from '$lib/constants';
	import { AttachmentType, MimeTypeApplication, PdfViewMode } from '$lib/enums';
	import type { ChatAttachmentDisplayItem } from '$lib/types';
	import { getLanguageFromFilename } from '$lib/utils';
	import { convertPDFToImage, convertPDFToText } from '$lib/utils/browser-only';

	interface Props {
		currentItem: ChatAttachmentDisplayItem | null;
		displayName: string;
		displayTextContent: string | undefined;
		hasVisionModality: boolean;
		activeModelId?: string;
	}

	let { activeModelId, currentItem, displayName, displayTextContent, hasVisionModality }: Props =
		$props();

	let pdfViewMode = $state<PdfViewMode>(PdfViewMode.PAGES);
	let pdfImages = $state<string[]>([]);
	let pdfImagesLoading = $state(false);
	// a failure stays on screen: the effect does not retry while an error is set
	let pdfImagesError = $state<string | null>(null);
	let pdfText = $state<string | null>(null);
	let pdfTextLoading = $state(false);
	let pdfTextError = $state<string | null>(null);

	let language = $derived(getLanguageFromFilename(displayName));

	// older PDFs sent as images stored a placeholder in `content`, so their text is extracted again
	let hasStoredText = $derived(
		currentItem?.attachment?.type === AttachmentType.PDF
			? !(
					currentItem.attachment.processedAsImages &&
					PDF_AS_IMAGES_LEGACY_CONTENT_REGEX.test(currentItem.attachment.content)
				)
			: !!displayTextContent
	);

	let isLoading = $derived(pdfImagesLoading || pdfTextLoading);

	function getPdfFile(): File | null {
		if (currentItem?.uploadedFile?.file) {
			return currentItem.uploadedFile.file;
		}

		if (
			currentItem?.attachment &&
			'base64Data' in currentItem.attachment &&
			currentItem.attachment.base64Data
		) {
			const byteArray = Uint8Array.from(atob(currentItem.attachment.base64Data), (char) =>
				char.charCodeAt(0)
			);

			return new File([byteArray], displayName, { type: MimeTypeApplication.PDF });
		}

		return null;
	}

	async function loadPdfImages() {
		if (pdfImages.length > 0 || pdfImagesLoading || !currentItem) return;

		if (
			currentItem.attachment?.type === AttachmentType.PDF &&
			currentItem.attachment.images?.length
		) {
			pdfImages = currentItem.attachment.images;

			return;
		}

		if (pdfImagesError) return;

		pdfImagesLoading = true;

		try {
			const file = getPdfFile();

			if (!file) {
				throw new Error('No PDF file available for conversion');
			}

			pdfImages = await convertPDFToImage(file);
		} catch (error) {
			pdfImagesError = error instanceof Error ? error.message : 'Failed to load PDF images';
		} finally {
			pdfImagesLoading = false;
		}
	}

	async function loadPdfText() {
		if (pdfText !== null || pdfTextLoading || !currentItem) return;

		if (hasStoredText) {
			pdfText = displayTextContent ?? '';

			return;
		}

		if (pdfTextError) return;

		pdfTextLoading = true;

		try {
			const file = getPdfFile();

			if (!file) {
				throw new Error('No PDF file available for text extraction');
			}

			pdfText = await convertPDFToText(file);
		} catch (error) {
			pdfTextError = error instanceof Error ? error.message : 'Failed to extract PDF text';
		} finally {
			pdfTextLoading = false;
		}
	}

	$effect(() => {
		if (pdfViewMode === PdfViewMode.PAGES) {
			loadPdfImages();
		} else {
			loadPdfText();
		}
	});
</script>

<div class="mb-4 flex items-center justify-end gap-2">
	<Button
		disabled={isLoading}
		onclick={() => (pdfViewMode = PdfViewMode.TEXT)}
		size="sm"
		variant={pdfViewMode === PdfViewMode.TEXT ? 'default' : 'outline'}
	>
		{#if pdfTextLoading}
			<div
				class="mr-1 {ICON_CLASS_DEFAULT} animate-spin rounded-full border-2 border-current border-t-transparent"
			></div>
		{:else}
			<FileText class="mr-1 {ICON_CLASS_DEFAULT}" />
		{/if}
		Text
	</Button>

	<Button
		disabled={isLoading}
		onclick={() => (pdfViewMode = PdfViewMode.PAGES)}
		size="sm"
		variant={pdfViewMode === PdfViewMode.PAGES ? 'default' : 'outline'}
	>
		{#if pdfImagesLoading}
			<div
				class="mr-1 {ICON_CLASS_DEFAULT} animate-spin rounded-full border-2 border-current border-t-transparent"
			></div>
		{:else}
			<Eye class="mr-1 {ICON_CLASS_DEFAULT}" />
		{/if}
		Pages
	</Button>
</div>

{#if !hasVisionModality && activeModelId && currentItem}
	<Alert.Root class="mb-4 max-w-4xl">
		<Info class={ICON_CLASS_DEFAULT} />

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

{#if pdfViewMode === PdfViewMode.TEXT}
	{#if pdfTextLoading}
		{@render loadingStatus('Extracting PDF text...')}
	{:else if pdfTextError}
		{@render status('Failed to extract PDF text', pdfTextError)}
	{:else if pdfText}
		<div class="px-4 pb-4">
			<SyntaxHighlightedCode class="max-w-4xl" code={pdfText} {language} maxHeight="none" />
		</div>
	{:else}
		{@render status('No text found in this PDF')}
	{/if}
{:else if pdfImagesLoading}
	{@render loadingStatus('Converting PDF to images...')}
{:else if pdfImagesError}
	{@render status('Failed to load PDF images', pdfImagesError)}
{:else if pdfImages.length > 0}
	{#each pdfImages as image, index (index)}
		<p class="mb-2 text-sm text-white/50">Page {index + 1}</p>

		<img alt="PDF Page {index + 1}" class="mx-auto max-w-[85vw] rounded-lg shadow-lg" src={image} />

		<div class="h-4"></div>
	{/each}
{:else}
	{@render status('No PDF pages available')}
{/if}

{#snippet loadingStatus(message: string)}
	<div class="flex flex-1 items-center justify-center p-8">
		<div class="text-center">
			<div
				class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent"
			></div>

			<p class="text-white/70">{message}</p>
		</div>
	</div>
{/snippet}

{#snippet status(message: string, detail?: string)}
	<div class="flex flex-1 items-center justify-center p-8">
		<div class="text-center">
			<FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />

			<p class={['text-white/70', detail && 'mb-4']}>{message}</p>

			{#if detail}
				<p class="text-sm text-white/50">{detail}</p>
			{/if}
		</div>
	</div>
{/snippet}
