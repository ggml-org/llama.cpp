<script lang="ts">
	import {
		ChatAttachmentThumbnailImage,
		ChatAttachmentThumbnailFile,
		HorizontalScrollCarousel,
		DialogChatAttachmentPreview,
		DialogChatAttachmentsViewAll
	} from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { getAttachmentDisplayItems } from '$lib/utils';

	interface Props {
		class?: string;
		style?: string;
		// For ChatMessage - stored attachments
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		// For ChatForm - pending uploads
		onFileRemove?: (fileId: string) => void;
		uploadedFiles?: ChatUploadedFile[];
		// Image size customization
		imageClass?: string;
		imageHeight?: string;
		imageWidth?: string;
		// Limit display to single row with "+ X more" button
		limitToSingleRow?: boolean;
		// For vision modality check
		activeModelId?: string;
	}

	let {
		class: className = '',
		style = '',
		attachments = [],
		readonly = false,
		onFileRemove,
		uploadedFiles = $bindable([]),
		// Default to small size for form previews
		imageClass = '',
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		limitToSingleRow = false,
		activeModelId
	}: Props = $props();

	let displayItems = $derived(getAttachmentDisplayItems({ uploadedFiles, attachments }));

	let carouselRef: HorizontalScrollCarousel | undefined = $state();
	let isScrollable = $state(false);
	let previewDialogOpen = $state(false);
	let previewItem = $state<ChatAttachmentPreviewItem | null>(null);
	let showViewAll = $derived(limitToSingleRow && displayItems.length > 0 && isScrollable);
	let viewAllDialogOpen = $state(false);

	function openPreview(item: ChatAttachmentDisplayItem, event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		previewItem = {
			uploadedFile: item.uploadedFile,
			attachment: item.attachment,
			preview: item.preview,
			name: item.name,
			size: item.size,
			textContent: item.textContent
		};
		previewDialogOpen = true;
	}

	$effect(() => {
		if (carouselRef && displayItems.length) {
			carouselRef.resetScroll();
		}
	});
</script>

{#if displayItems.length > 0}
	<div class={className} {style}>
		{#if limitToSingleRow}
			<HorizontalScrollCarousel
				bind:this={carouselRef}
				onScrollableChange={(scrollable) => (isScrollable = scrollable)}
			>
				{#each displayItems as item (item.id)}
					{#if item.isImage && item.preview}
						<ChatAttachmentThumbnailImage
							class="flex-shrink-0 cursor-pointer {limitToSingleRow ? 'first:ml-4 last:mr-4' : ''}"
							id={item.id}
							name={item.name}
							preview={item.preview}
							{readonly}
							onRemove={onFileRemove}
							height={imageHeight}
							width={imageWidth}
							{imageClass}
							onClick={(event) => openPreview(item, event)}
						/>
					{:else}
						<ChatAttachmentThumbnailFile
							class="flex-shrink-0 cursor-pointer {limitToSingleRow ? 'first:ml-4 last:mr-4' : ''}"
							id={item.id}
							name={item.name}
							size={item.size}
							{readonly}
							onRemove={onFileRemove}
							textContent={item.textContent}
							attachment={item.attachment}
							uploadedFile={item.uploadedFile}
							onClick={(event) => openPreview(item, event)}
						/>
					{/if}
				{/each}
			</HorizontalScrollCarousel>

			{#if showViewAll}
				<div class="mt-2 -mr-2 flex justify-end px-4">
					<Button
						type="button"
						variant="ghost"
						size="sm"
						class="h-6 text-xs text-muted-foreground hover:text-foreground"
						onclick={() => (viewAllDialogOpen = true)}
					>
						View all ({displayItems.length})
					</Button>
				</div>
			{/if}
		{:else}
			<div class="flex flex-wrap items-start justify-end gap-3">
				{#each displayItems as item (item.id)}
					{#if item.isImage && item.preview}
						<ChatAttachmentThumbnailImage
							class="cursor-pointer"
							id={item.id}
							name={item.name}
							preview={item.preview}
							{readonly}
							onRemove={onFileRemove}
							height={imageHeight}
							width={imageWidth}
							{imageClass}
							onClick={(event) => openPreview(item, event)}
						/>
					{:else}
						<ChatAttachmentThumbnailFile
							class="cursor-pointer"
							id={item.id}
							name={item.name}
							size={item.size}
							{readonly}
							onRemove={onFileRemove}
							textContent={item.textContent}
							attachment={item.attachment}
							uploadedFile={item.uploadedFile}
							onClick={(event?: MouseEvent) => openPreview(item, event)}
						/>
					{/if}
				{/each}
			</div>
		{/if}
	</div>
{/if}

{#if previewItem}
	<DialogChatAttachmentPreview
		bind:open={previewDialogOpen}
		uploadedFile={previewItem.uploadedFile}
		attachment={previewItem.attachment}
		preview={previewItem.preview}
		name={previewItem.name}
		size={previewItem.size}
		textContent={previewItem.textContent}
		{activeModelId}
	/>
{/if}

<DialogChatAttachmentsViewAll
	bind:open={viewAllDialogOpen}
	{uploadedFiles}
	{attachments}
	{readonly}
	{onFileRemove}
	imageHeight="h-64"
	{imageClass}
	{activeModelId}
/>
