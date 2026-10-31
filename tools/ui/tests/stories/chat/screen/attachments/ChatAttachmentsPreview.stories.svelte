<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatAttachmentsPreview from '$lib/components/app/chat/ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreview.svelte';
	import { AttachmentType } from '$lib/enums';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatAttachmentsPreview,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Components/ChatScreen/ChatAttachments/ChatAttachmentsPreview'
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
