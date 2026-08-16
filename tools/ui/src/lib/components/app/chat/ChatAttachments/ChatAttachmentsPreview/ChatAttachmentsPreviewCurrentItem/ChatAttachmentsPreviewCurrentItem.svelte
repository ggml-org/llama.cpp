<script lang="ts">
	import ChatAttachmentsPreviewCurrentItemAudio from './ChatAttachmentsPreviewCurrentItemAudio.svelte';
	import ChatAttachmentsPreviewCurrentItemImage from './ChatAttachmentsPreviewCurrentItemImage.svelte';
	import ChatAttachmentsPreviewCurrentItemPdf from './ChatAttachmentsPreviewCurrentItemPdf.svelte';
	import ChatAttachmentsPreviewCurrentItemText from './ChatAttachmentsPreviewCurrentItemText.svelte';
	import ChatAttachmentsPreviewCurrentItemUnavailable from './ChatAttachmentsPreviewCurrentItemUnavailable.svelte';
	import ChatAttachmentsPreviewCurrentItemVideo from './ChatAttachmentsPreviewCurrentItemVideo.svelte';
	import { FileIcon, FileText, Image, Music, Video } from '@lucide/svelte';
	import type { ChatAttachmentDisplayItem } from '$lib/types';

	interface Props {
		currentItem: ChatAttachmentDisplayItem | null;
		isImage: boolean;
		isAudio: boolean;
		isVideo: boolean;
		isPdf: boolean;
		isText: boolean;
		displayPreview: string | undefined;
		displayTextContent: string | undefined;
		audioSrc: string | null;
		videoSrc: string | null;
		language: string;
		hasVisionModality: boolean;
		activeModelId?: string;
	}

	let {
		activeModelId,
		audioSrc,
		currentItem,
		displayPreview,
		displayTextContent,
		hasVisionModality,
		isAudio,
		isImage,
		isPdf,
		isText,
		isVideo,
		language,
		videoSrc
	}: Props = $props();

	let IconComponent = $derived(
		isImage ? Image : isText || isPdf ? FileText : isAudio ? Music : isVideo ? Video : FileIcon
	);

	let isUnavailable = $derived(
		!isPdf && !isImage && !(isText && displayTextContent) && !isAudio && !isVideo
	);
</script>

{#if currentItem}
	{#key currentItem.id}
		{#if isPdf}
			<ChatAttachmentsPreviewCurrentItemPdf
				{currentItem}
				displayName={currentItem.name}
				{displayTextContent}
				{hasVisionModality}
				{activeModelId}
			/>
		{:else if isImage}
			<ChatAttachmentsPreviewCurrentItemImage {currentItem} {displayPreview} />
		{:else if isText && displayTextContent}
			<ChatAttachmentsPreviewCurrentItemText {displayTextContent} {language} />
		{:else if isAudio}
			<ChatAttachmentsPreviewCurrentItemAudio {currentItem} {audioSrc} />
		{:else if isVideo}
			<ChatAttachmentsPreviewCurrentItemVideo {currentItem} {videoSrc} />
		{:else if isUnavailable}
			<ChatAttachmentsPreviewCurrentItemUnavailable {IconComponent} />
		{/if}
	{/key}
{/if}
