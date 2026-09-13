<script lang="ts">
	import { Trash2 } from '@lucide/svelte';
	import {
		DialogChatError,
		DialogConfirmation,
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

<DialogConfirmation
	bind:open={showDeleteDialog}
	cancelText="Cancel"
	confirmText="Delete"
	description="Are you sure you want to delete this conversation? This action cannot be undone and will permanently remove all messages in this conversation."
	icon={Trash2}
	onCancel={() => (showDeleteDialog = false)}
	onConfirm={handleDeleteConfirm}
	title="Delete Conversation"
	variant="destructive"
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
