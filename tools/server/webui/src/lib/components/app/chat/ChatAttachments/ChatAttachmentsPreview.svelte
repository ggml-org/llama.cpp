<script lang="ts">
	import {
		ChatAttachmentsPreviewCurrentItem,
		ChatAttachmentsPreviewFileInfo,
		ChatAttachmentsPreviewNavButtons,
		ChatAttachmentsPreviewThumbnailStrip
	} from '$lib/components/app';
	import { modelsStore } from '$lib/stores/models.svelte';
	import {
		createBase64DataUrl,
		formatFileSize,
		getAttachmentDisplayItems,
		getLanguageFromFilename,
		isAudioFile,
		isImageFile,
		isMcpPrompt,
		isMcpResource,
		isPdfFile,
		isTextFile
	} from '$lib/utils';

	interface PreviewItem {
		id: string;
		name: string;
		size?: number;
		preview?: string;
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		textContent?: string;
		isImage: boolean;
		isAudio: boolean;
	}

	interface Props {
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		activeModelId?: string;
		class?: string;
		previewFocusIndex?: number;
	}

	let {
		uploadedFiles = [],
		attachments = [],
		activeModelId,
		class: className = '',
		previewFocusIndex = 0
	}: Props = $props();

	let allItems = $derived(
		getAttachmentDisplayItems({ uploadedFiles, attachments })
			.filter((item) => !isMcpPrompt(item) && !isMcpResource(item))
			.map(
				(item): PreviewItem => ({
					...item,
					isImage: isImageFile(item.attachment, item.uploadedFile),
					isAudio: isAudioFile(item.attachment, item.uploadedFile)
				})
			)
	);

	let currentIndex = $state(0);

	$effect(() => {
		if (previewFocusIndex >= 0 && previewFocusIndex < allItems.length) {
			currentIndex = previewFocusIndex;
		}
	});

	$effect(() => {
		const handler = (e: Event) => {
			const delta = (e as CustomEvent).detail;

			if (delta < 0) {
				currentIndex = currentIndex > 0 ? currentIndex - 1 : allItems.length - 1;
			} else {
				currentIndex = currentIndex < allItems.length - 1 ? currentIndex + 1 : 0;
			}
		};

		document.addEventListener('chat-attachments-nav', handler);

		return () => document.removeEventListener('chat-attachments-nav', handler);
	});

	$effect(() => {
		const index = currentIndex;
		setTimeout(() => {
			const thumbnail = document.querySelector(`[data-thumbnail-index="${index}"]`);

			thumbnail?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
		}, 0);
	});

	let currentItem = $derived(allItems[currentIndex] ?? null);
	let displayName = $derived(
		currentItem?.name ||
			currentItem?.uploadedFile?.name ||
			currentItem?.attachment?.name ||
			'Unknown File'
	);
	let isAudio = $derived(
		currentItem ? isAudioFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isImage = $derived(
		currentItem ? isImageFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isPdf = $derived(
		currentItem ? isPdfFile(currentItem.attachment, currentItem.uploadedFile) : false
	);
	let isText = $derived(
		currentItem ? isTextFile(currentItem.attachment, currentItem.uploadedFile) : false
	);

	let displayPreview = $derived(
		currentItem?.uploadedFile?.preview ||
			(isImage && currentItem?.attachment && 'base64Url' in currentItem.attachment
				? currentItem.attachment.base64Url
				: currentItem?.preview)
	);

	let displayTextContent = $derived(
		currentItem?.uploadedFile?.textContent ||
			(currentItem?.attachment && 'content' in currentItem.attachment
				? currentItem.attachment.content
				: currentItem?.textContent)
	);

	let language = $derived(getLanguageFromFilename(displayName));

	let fileSize = $derived(currentItem?.size ? formatFileSize(currentItem.size) : '');

	let hasVisionModality = $derived(
		currentItem && activeModelId ? modelsStore.modelSupportsVision(activeModelId) : false
	);

	let audioSrc = $derived(
		isAudio && currentItem
			? (currentItem.uploadedFile?.preview ??
					(currentItem.attachment &&
					'mimeType' in currentItem.attachment &&
					'base64Data' in currentItem.attachment
						? createBase64DataUrl(
								currentItem.attachment.mimeType,
								currentItem.attachment.base64Data
							)
						: null))
			: null
	);

	export function prev() {
		currentIndex = currentIndex > 0 ? currentIndex - 1 : allItems.length - 1;
	}

	export function next() {
		currentIndex = currentIndex < allItems.length - 1 ? currentIndex + 1 : 0;
	}

	function onNavigate(index: number) {
		currentIndex = index;
	}
</script>

<div class="{className} flex flex-col text-white">
	<div class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
		<ChatAttachmentsPreviewNavButtons onPrev={prev} onNext={next} show={allItems.length > 1} />

		<div class="flex h-full w-full flex-col items-center justify-start overflow-auto py-4">
			{#if currentItem}
				<ChatAttachmentsPreviewFileInfo {displayName} {fileSize} />

				<ChatAttachmentsPreviewCurrentItem
					{currentItem}
					{isImage}
					{isAudio}
					{isPdf}
					{isText}
					{displayPreview}
					{displayTextContent}
					{audioSrc}
					{language}
					{hasVisionModality}
					{activeModelId}
				/>
			{/if}

			<ChatAttachmentsPreviewThumbnailStrip items={allItems} {currentIndex} {onNavigate} />
		</div>
	</div>
</div>
