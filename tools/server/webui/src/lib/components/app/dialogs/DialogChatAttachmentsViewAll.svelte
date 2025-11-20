<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ChatAttachmentsViewAll } from '$lib/components/app';

	interface Props {
		open?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		onFileRemove?: (fileId: string) => void;
		imageHeight?: string;
		imageWidth?: string;
		imageClass?: string;
	}

	let {
		open = $bindable(false),
		uploadedFiles = [],
		attachments = [],
		readonly = false,
		onFileRemove,
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		imageClass = ''
	}: Props = $props();

	let totalCount = $derived(uploadedFiles.length + attachments.length);
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay />

		<Dialog.Content class="flex !max-h-[90vh] !max-w-6xl flex-col">
			<Dialog.Header>
				<Dialog.Title>All Attachments ({totalCount})</Dialog.Title>
				<Dialog.Description>View and manage all attached files</Dialog.Description>
			</Dialog.Header>

			<ChatAttachmentsViewAll
				{uploadedFiles}
				{attachments}
				{readonly}
				{onFileRemove}
				{imageHeight}
				{imageWidth}
				{imageClass}
			/>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
