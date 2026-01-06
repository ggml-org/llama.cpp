<script lang="ts">
	import { ChatAttachmentThumbnailImage, ChatAttachmentThumbnailFile } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { DialogChatAttachmentPreview, DialogChatAttachmentsViewAll } from '$lib/components/app';
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

	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let isScrollable = $state(false);
	let previewDialogOpen = $state(false);
	let previewItem = $state<ChatAttachmentPreviewItem | null>(null);
	let scrollContainer: HTMLDivElement | undefined = $state();
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

	function scrollLeft(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: scrollContainer.clientWidth * -0.67, behavior: 'smooth' });
	}

	function scrollRight(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: scrollContainer.clientWidth * 0.67, behavior: 'smooth' });
	}

	function updateScrollButtons() {
		if (!scrollContainer) return;

		const { scrollLeft, scrollWidth, clientWidth } = scrollContainer;

		canScrollLeft = scrollLeft > 0;
		canScrollRight = scrollLeft < scrollWidth - clientWidth - 1;
		isScrollable = scrollWidth > clientWidth;
	}

	$effect(() => {
		if (scrollContainer && displayItems.length) {
			scrollContainer.scrollLeft = 0;

			setTimeout(() => {
				updateScrollButtons();
			}, 0);
		}
	});
</script>

{#if displayItems.length > 0}
	<div class={className} {style}>
		{#if limitToSingleRow}
			<div class="relative">
				<button
					class="absolute top-1/2 left-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-foreground/15 shadow-md backdrop-blur-xs transition-opacity hover:bg-foreground/35 {canScrollLeft
						? 'opacity-100'
						: 'pointer-events-none opacity-0'}"
					onclick={scrollLeft}
					aria-label="Scroll left"
				>
					<ChevronLeft class="h-4 w-4" />
				</button>

				<div
					class="scrollbar-hide flex items-start gap-3 overflow-x-auto"
					bind:this={scrollContainer}
					onscroll={updateScrollButtons}
				>
					{#each displayItems as item (item.id)}
						{#if item.isImage && item.preview}
							<ChatAttachmentThumbnailImage
								class="flex-shrink-0 cursor-pointer {limitToSingleRow
									? 'first:ml-4 last:mr-4'
									: ''}"
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
								class="flex-shrink-0 cursor-pointer {limitToSingleRow
									? 'first:ml-4 last:mr-4'
									: ''}"
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
				</div>

				<button
					class="absolute top-1/2 right-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-foreground/15 shadow-md backdrop-blur-xs transition-opacity hover:bg-foreground/35 {canScrollRight
						? 'opacity-100'
						: 'pointer-events-none opacity-0'}"
					onclick={scrollRight}
					aria-label="Scroll right"
				>
					<ChevronRight class="h-4 w-4" />
				</button>
			</div>

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
