<script lang="ts">
	import {
		DialogChatAttachmentsPreview,
		DialogChatError,
		DialogConversationDelete,
		DialogEmptyFileAlert,
		DialogFileUploadError
	} from '#lib/components/app/index.ts';
	import { ErrorDialogType } from '#lib/enums/index.ts';
	import { mediaGalleryStore } from '#lib/stores/index.ts';

	let {
		activeErrorDialog,
		emptyFileNames,
		fileUpload,
		handleDeleteConfirm,
		handleErrorDialogOpenChange,
		showDeleteDialog,
		showEmptyFileDialog
	} = $props();
</script>

<DialogFileUploadError
	bind:open={fileUpload.showFileErrorDialog}
	fileErrorData={fileUpload.fileErrorData}
/>

<DialogConversationDelete
	bind:open={showDeleteDialog}
	onCancel={() => (showDeleteDialog = false)}
	onConfirm={handleDeleteConfirm}
/>

<DialogEmptyFileAlert
	bind:open={showEmptyFileDialog}
	emptyFiles={emptyFileNames}
	onOpenChange={(open) => {
		if (!open) {
			emptyFileNames = [];
		}
	}}
/>

<DialogChatError
	contextInfo={activeErrorDialog?.contextInfo}
	message={activeErrorDialog?.message ?? ''}
	onOpenChange={handleErrorDialogOpenChange}
	open={Boolean(activeErrorDialog)}
	type={activeErrorDialog?.type ?? ErrorDialogType.SERVER}
/>

<!-- the conversation's media as one slideshow, opened from any thumbnail in the chat -->
<DialogChatAttachmentsPreview
	bind:open={mediaGalleryStore.open}
	attachments={mediaGalleryStore.items}
	previewFocusIndex={mediaGalleryStore.index}
/>
