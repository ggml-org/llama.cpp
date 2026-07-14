<script lang="ts">
	import {
		ChatAttachmentsListItem,
		DialogChatAttachmentsPreview,
		DialogMcpResourcePreview,
		HorizontalScrollCarousel
	} from '$lib/components/app';
	import type { DatabaseMessageExtraMcpResource } from '$lib/types';
	import { getAttachmentDisplayItems, isMcpPrompt, isMcpResource } from '$lib/utils';

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

	let carouselRef: HorizontalScrollCarousel | undefined = $state();
	let mcpResourcePreviewOpen = $state(false);
	let mcpResourcePreviewExtra = $state<DatabaseMessageExtraMcpResource | null>(null);
	let previewFocusIndex = $state(0);
	let viewAllDialogOpen = $state(false);

	let displayItems = $derived(getAttachmentDisplayItems({ uploadedFiles, attachments }));

	function openPreview(item: ChatAttachmentDisplayItem, event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		// Find the index of the clicked item among non-MCP attachments
		const nonMcpItems = displayItems.filter((i) => !isMcpPrompt(i) && !isMcpResource(i));
		const index = nonMcpItems.findIndex((i) => i.id === item.id);

		previewFocusIndex = index >= 0 ? index : 0;
		viewAllDialogOpen = true;
	}

	function openMcpResourcePreview(extra: DatabaseMessageExtraMcpResource) {
		mcpResourcePreviewExtra = extra;
		mcpResourcePreviewOpen = true;
	}

	$effect(() => {
		if (carouselRef && displayItems.length) {
			carouselRef.resetScroll();
		}
	});
</script>

{#snippet attachmentitem(item: ChatAttachmentDisplayItem)}
	<ChatAttachmentsListItem
		{imageClass}
		{imageHeight}
		{imageWidth}
		{item}
		{limitToSingleRow}
		{onFileRemove}
		onMcpResourcePreview={openMcpResourcePreview}
		onPreview={(i: ChatAttachmentDisplayItem, event?: MouseEvent) => openPreview(i, event)}
		{readonly}
	/>
{/snippet}

{#if displayItems.length > 0}
	<div class={className} {style}>
		{#if limitToSingleRow}
			<HorizontalScrollCarousel bind:this={carouselRef}>
				{#each displayItems as item (item.id)}
					{@render attachmentitem(item)}
				{/each}
			</HorizontalScrollCarousel>
		{:else}
			<div class="flex flex-wrap items-start justify-end gap-3">
				{#each displayItems as item (item.id)}
					{@render attachmentitem(item)}
				{/each}
			</div>
		{/if}
	</div>
{/if}

<DialogChatAttachmentsPreview
	{activeModelId}
	{attachments}
	bind:open={viewAllDialogOpen}
	{previewFocusIndex}
	{uploadedFiles}
/>

{#if mcpResourcePreviewExtra}
	<DialogMcpResourcePreview extra={mcpResourcePreviewExtra} bind:open={mcpResourcePreviewOpen} />
{/if}
