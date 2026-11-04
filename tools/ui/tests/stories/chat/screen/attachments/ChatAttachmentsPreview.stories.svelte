<script lang="ts" module>
	import {
		gradientAttachment,
		imageAttachment,
		logoAttachment,
		pdfAttachment,
		textAttachment,
		uploadedArchive,
		uploadedAudio,
		uploadedVideo
	} from '../fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatAttachmentsPreview from '$lib/components/app/chat/ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreview.svelte';
	import { AttachmentType } from '$lib/enums';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatAttachmentsPreview,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Blocks/Chat/ChatScreen/ChatAttachments/ChatAttachmentsPreview'
	});

	// same backdrop as DialogChatAttachmentsPreview
	const frameClass = 'flex h-screen flex-col bg-neutral-950';

	// valid base64 that is not a PDF: pdf.js throws InvalidPDFException
	const brokenPdfAttachment: DatabaseMessageExtraPdfFile = {
		base64Data: btoa('this is not a pdf'),
		content: 'Extracted text of the broken PDF.',
		name: 'broken.pdf',
		processedAsImages: false,
		size: 17,
		type: AttachmentType.PDF
	};
</script>

<!-- several items: nav buttons and thumbnail strip are shown -->
<Story asChild name="Default">
	<div class={frameClass}>
		<ChatAttachmentsPreview
			attachments={[imageAttachment, logoAttachment, gradientAttachment, textAttachment]}
			class="min-h-0 flex-1"
			previewFocusIndex={1}
			uploadedFiles={[uploadedAudio, uploadedVideo]}
		/>
	</div>
</Story>

<Story asChild name="Image">
	<div class={frameClass}>
		<ChatAttachmentsPreview attachments={[imageAttachment]} class="min-h-0 flex-1" />
	</div>
</Story>

<Story asChild name="Text">
	<div class={frameClass}>
		<ChatAttachmentsPreview attachments={[textAttachment]} class="min-h-0 flex-1" />
	</div>
</Story>

<!-- pages come from the pre-rendered images of the stored attachment -->
<Story asChild name="Pdf">
	<div class={frameClass}>
		<ChatAttachmentsPreview attachments={[pdfAttachment]} class="min-h-0 flex-1" />
	</div>
</Story>

<!-- unknown model has no vision modality: shows the "Preview only" alert -->
<Story asChild name="PdfWithoutVision">
	<div class={frameClass}>
		<ChatAttachmentsPreview
			activeModelId="text-only-model"
			attachments={[pdfAttachment]}
			class="min-h-0 flex-1"
		/>
	</div>
</Story>

<Story asChild name="Audio">
	<div class={frameClass}>
		<ChatAttachmentsPreview class="min-h-0 flex-1" uploadedFiles={[uploadedAudio]} />
	</div>
</Story>

<!-- no preview URL: no player -->
<Story asChild name="AudioUnavailable">
	<div class={frameClass}>
		<ChatAttachmentsPreview
			class="min-h-0 flex-1"
			uploadedFiles={[{ ...uploadedAudio, preview: undefined }]}
		/>
	</div>
</Story>

<Story asChild name="Video">
	<div class={frameClass}>
		<ChatAttachmentsPreview class="min-h-0 flex-1" uploadedFiles={[uploadedVideo]} />
	</div>
</Story>

<Story asChild name="VideoUnavailable">
	<div class={frameClass}>
		<ChatAttachmentsPreview
			class="min-h-0 flex-1"
			uploadedFiles={[{ ...uploadedVideo, preview: undefined }]}
		/>
	</div>
</Story>

<Story asChild name="Unavailable">
	<div class={frameClass}>
		<ChatAttachmentsPreview class="min-h-0 flex-1" uploadedFiles={[uploadedArchive]} />
	</div>
</Story>

<!-- conversion fails once: shows the error and does not retry -->
<Story
	asChild
	name="PdfConversionError"
	play={async ({ canvas }) => {
		await waitFor(() => expect(canvas.getByText('Failed to load PDF images')).toBeInTheDocument(), {
			timeout: 5000
		});

		// a retry loop would flip back to the loading state
		await new Promise((resolve) => setTimeout(resolve, 500));
		await expect(canvas.queryByText('Converting PDF to images...')).not.toBeInTheDocument();
		await expect(canvas.getByText('Failed to load PDF images')).toBeInTheDocument();
	}}
>
	<div class={frameClass}>
		<ChatAttachmentsPreview attachments={[brokenPdfAttachment]} class="min-h-0 flex-1" />
	</div>
</Story>
