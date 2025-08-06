<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { FileText, Image, Music, FileIcon } from '@lucide/svelte';
	import type { DatabaseMessageExtra } from '$lib/types/database.d.ts';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';

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

	let displayName = $derived(uploadedFile?.name || attachment?.name || name || 'Unknown File');

	let displayType = $derived(
		uploadedFile?.type ||
			(attachment?.type === 'imageFile'
				? 'image'
				: attachment?.type === 'textFile'
					? 'text'
					: attachment?.type === 'audioFile'
						? (attachment as any).mimeType || 'audio'
						: attachment?.type === 'pdfFile'
							? 'application/pdf'
							: type || 'unknown')
	);

	let displaySize = $derived(uploadedFile?.size || size);

	let displayPreview = $derived(
		uploadedFile?.preview ||
			(attachment?.type === 'imageFile' ? (attachment as any).base64Url : preview)
	);

	let displayTextContent = $derived(
		uploadedFile?.textContent ||
			(attachment?.type === 'textFile'
				? (attachment as any).content
				: attachment?.type === 'pdfFile'
					? (attachment as any).content
					: textContent)
	);

	let isImage = $derived(displayType.startsWith('image/') || displayType === 'image');
	let isText = $derived(displayType.startsWith('text/') || displayType === 'text');
	let isPdf = $derived(displayType === 'application/pdf');
	let isAudio = $derived(displayType.startsWith('audio/') || displayType === 'audio');

	let IconComponent = $derived(() => {
		if (isImage) return Image;
		if (isText || isPdf) return FileText;
		if (isAudio) return Music;
		return FileIcon;
	});

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';

		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));

		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
</script>

<Dialog.Root bind:open>
	<Dialog.Content class="grid max-h-[90vh] max-w-4xl overflow-hidden sm:w-auto sm:max-w-6xl">
		<Dialog.Header class="flex-shrink-0">
			<div class="flex items-center space-x-4">
				<div class="flex items-center gap-3">
					{#if IconComponent}
						<IconComponent class="text-muted-foreground h-5 w-5" />
					{/if}

					<div>
						<Dialog.Title class="text-left">{displayName}</Dialog.Title>

						<div class="text-muted-foreground flex items-center gap-2 text-sm">
							<span>{displayType}</span>
							{#if displaySize}
								<span>•</span>
								<span>{formatFileSize(displaySize)}</span>
							{/if}
						</div>
					</div>
				</div>
			</div>
		</Dialog.Header>

		<div class="flex-1 overflow-auto">
			{#if isImage && displayPreview}
				<div class="flex items-center justify-center p-4">
					<img
						src={displayPreview}
						alt={displayName}
						class="max-h-full rounded-lg object-contain shadow-lg"
					/>
				</div>
			{:else if (isText || isPdf) && displayTextContent}
				<div class="p-4">
					<div
						class="bg-muted max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-lg p-4 font-mono text-sm"
					>
						{displayTextContent}
					</div>
				</div>
			{:else if isAudio && attachment?.type === 'audioFile'}
				<div class="flex items-center justify-center p-8">
					<div class="text-center">
						<Music class="text-muted-foreground mx-auto mb-4 h-16 w-16" />

						<p class="text-muted-foreground mb-4">Audio file preview not available</p>
					</div>
				</div>
			{:else}
				<div class="flex items-center justify-center p-8">
					<div class="text-center">
						{#if IconComponent}
							<IconComponent class="text-muted-foreground mx-auto mb-4 h-16 w-16" />
						{/if}

						<p class="text-muted-foreground mb-4">
							Preview not available for this file type
						</p>
					</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
