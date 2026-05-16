<script lang="ts">
	import type { ChatAttachmentDisplayItem } from '$lib/types';
	import { Image, Music, FileText, FileIcon } from '@lucide/svelte';
	import ChatAttachmentsPreviewCurrentItemPdf from './ChatAttachmentsPreviewCurrentItemPdf.svelte';
	import ChatAttachmentsPreviewCurrentItemImage from './ChatAttachmentsPreviewCurrentItemImage.svelte';
	import ChatAttachmentsPreviewCurrentItemAudio from './ChatAttachmentsPreviewCurrentItemAudio.svelte';
	import ChatAttachmentsPreviewCurrentItemText from './ChatAttachmentsPreviewCurrentItemText.svelte';
	import ChatAttachmentsPreviewCurrentItemUnavailable from './ChatAttachmentsPreviewCurrentItemUnavailable.svelte';

	interface Props {
		currentItem: ChatAttachmentDisplayItem | null;
		isImage: boolean;
		isAudio: boolean;
		isPdf: boolean;
		isText: boolean;
		displayPreview: string | undefined;
		displayTextContent: string | undefined;
		audioSrc: string | null;
		language: string;
		hasVisionModality: boolean;
		activeModelId?: string;
	}

	let {
		currentItem,
		isImage,
		isAudio,
		isPdf,
		isText,
		displayPreview,
		displayTextContent,
		audioSrc,
		language,
		hasVisionModality,
		activeModelId
	}: Props = $props();

	let IconComponent = $derived(
		isImage ? Image : isText || isPdf ? FileText : isAudio ? Music : FileIcon
	);

	let isUnavailable = $derived(!isPdf && !isImage && !(isText && displayTextContent) && !isAudio);
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
		{:else if isUnavailable}
			<ChatAttachmentsPreviewCurrentItemUnavailable {IconComponent} />
		{/if}
	{/key}
{/if}
