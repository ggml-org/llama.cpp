<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ChatAttachmentsPreview } from '$lib/components/app';
	import { formatFileSize, getAttachmentDisplayItems } from '$lib/utils';

	interface Props {
		open: boolean;
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		activeModelId?: string;
		previewFocusIndex?: number;
	}

	let {
		open = $bindable(false),
		uploadedFiles = [],
		attachments = [],
		activeModelId,
		previewFocusIndex = 0
	}: Props = $props();

	let displayItems = $derived(
		getAttachmentDisplayItems({ uploadedFiles, attachments }).filter(
			(item) => !item.isMcpPrompt && !item.isMcpResource
		)
	);
	let totalCount = $derived(displayItems.length);

	let displayName = $derived(
		displayItems.length === 1
			? (displayItems[0]?.name ?? 'Attachment')
			: `Attachments (${totalCount})`
	);
	let displaySize = $derived(displayItems.length === 1 ? displayItems[0]?.size : undefined);
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay />
		<Dialog.Content class="flex !max-h-[90vh] !max-w-6xl flex-col">
			<Dialog.Header>
				<Dialog.Title class="pr-8">{displayName}</Dialog.Title>
				<Dialog.Description>
					{#if displaySize}
						{formatFileSize(displaySize)}
					{:else}
						{totalCount} attachment{totalCount !== 1 ? 's' : ''}
					{/if}
				</Dialog.Description>
			</Dialog.Header>
			<ChatAttachmentsPreview
				{uploadedFiles}
				{attachments}
				{activeModelId}
				{previewFocusIndex}
				class="min-h-0 flex-1 overflow-y-auto px-1"
			/>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
