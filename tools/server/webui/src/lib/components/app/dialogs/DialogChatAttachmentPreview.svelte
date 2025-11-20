<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ChatAttachmentPreview } from '$lib/components/app';

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

	let chatAttachmentPreviewRef: ChatAttachmentPreview | undefined = $state();

	$effect(() => {
		if (open && chatAttachmentPreviewRef) {
			chatAttachmentPreviewRef.reset();
		}
	});
</script>

<Dialog.Root bind:open>
	<Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden !p-10 sm:w-auto sm:max-w-6xl">
		<ChatAttachmentPreview
			bind:this={chatAttachmentPreviewRef}
			{uploadedFile}
			{attachment}
			{preview}
			{name}
			{type}
			{size}
			{textContent}
		/>
	</Dialog.Content>
</Dialog.Root>
