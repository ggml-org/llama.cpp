<script lang="ts">
	import {
		DialogChatError,
		DialogConversationDelete,
		DialogEmptyFileAlert,
		DialogFileUploadError
	} from '$lib/components/app';
	import { ErrorDialogType } from '$lib/enums';

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
