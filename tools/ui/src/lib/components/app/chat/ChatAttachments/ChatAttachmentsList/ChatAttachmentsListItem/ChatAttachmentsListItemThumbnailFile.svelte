<script lang="ts">
	import { X } from '@lucide/svelte';
	import {
		formatFileSize,
		getFileTypeLabel,
		getPreviewText,
		isPdfFile,
		isTextFile
	} from '$lib/utils';
	import { ActionIcon } from '$lib/components/app';
	import { AttachmentType } from '$lib/enums';

	interface Props {
		attachment?: DatabaseMessageExtra;
		class?: string;
		id: string;
		onclick?: (event: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		readonly?: boolean;
		size?: number;
		textContent?: string;
		// Either uploaded file or stored attachment
		uploadedFile?: ChatUploadedFile;
	}

	let {
		attachment,
		class: className = '',
		id,
		onclick,
		onRemove,
		name,
		readonly = false,
		size,
		textContent,
		uploadedFile
	}: Props = $props();

	let isPdf = $derived(isPdfFile(attachment, uploadedFile));
	let isPdfWithContent = $derived(isPdf && !!textContent);

	let isText = $derived(isTextFile(attachment, uploadedFile));
	let isTextWithContent = $derived(isText && !!textContent);

	let fileTypeLabel = $derived.by(() => {
		if (uploadedFile?.type) {
			return getFileTypeLabel(uploadedFile.type);
		}

		if (attachment) {
			if ('mimeType' in attachment && attachment.mimeType) {
				return getFileTypeLabel(attachment.mimeType);
			}

			if (attachment.type) {
				return getFileTypeLabel(attachment.type);
			}
		}

		return getFileTypeLabel(name);
	});

	let pdfProcessingMode = $derived.by(() => {
		if (attachment?.type === AttachmentType.PDF) {
			const pdfAttachment = attachment as DatabaseMessageExtraPdfFile;

			return pdfAttachment.processedAsImages ? 'Sent as Image' : 'Sent as Text';
		}

		return null;
	});
</script>

{#snippet textPreview(content: string)}
	<div class="relative">
		<div
			class="font-mono text-xs leading-relaxed break-words whitespace-pre-wrap text-muted-foreground {!readonly
				? 'max-h-3rem line-height-1.2'
				: ''}"
		>
			{getPreviewText(content)}
		</div>

		{#if content.length > 150}
			<div
				class="pointer-events-none absolute right-0 bottom-0 left-0 h-4 bg-gradient-to-t from-muted to-transparent {readonly
					? 'h-6'
					: ''}"
			></div>
		{/if}
	</div>
{/snippet}

{#snippet removeButton()}
	<div class="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100">
		<ActionIcon icon={X} tooltip="Remove" stopPropagationOnClick onclick={() => onRemove?.(id)} />
	</div>
{/snippet}

{#snippet fileIcon()}
	<div
		class="flex h-8 w-8 items-center justify-center rounded bg-primary/10 text-xs font-medium text-primary"
	>
		{fileTypeLabel}
	</div>
{/snippet}

{#snippet info(text: string | undefined)}
	{#if text}
		<span class="text-xs text-muted-foreground">{text}</span>
	{/if}
{/snippet}

{#if isTextWithContent || isPdfWithContent}
	<button
		aria-label={readonly ? `Preview ${name}` : undefined}
		class="rounded-lg border border-border bg-muted p-3 {className} cursor-pointer {readonly
			? 'w-full max-w-2xl transition-shadow hover:shadow-md'
			: `group relative text-left ${textContent ? 'max-h-24 max-w-72' : 'max-w-36'}`} overflow-hidden"
		{onclick}
		type="button"
	>
		{#if !readonly}
			{@render removeButton()}
		{/if}

		<div class={[!readonly && 'pr-8', 'overflow-hidden']}>
			{#if readonly}
				<div class="flex items-start gap-3">
					<div class="flex min-w-0 flex-1 flex-col items-start text-left">
						<span class="w-full truncate text-sm font-medium text-foreground">{name}</span>

						{@render info(pdfProcessingMode || (size ? formatFileSize(size) : undefined))}

						{#if textContent}
							{@render textPreview(textContent)}
						{/if}
					</div>
				</div>
			{:else}
				<span class="mb-3 block truncate text-sm font-medium text-foreground">{name}</span>

				{#if textContent}
					{@render textPreview(textContent)}
				{/if}
			{/if}
		</div>
	</button>
{:else}
	<button
		class="group flex items-center gap-3 rounded-lg border border-border bg-muted p-3 {className} relative"
		{onclick}
		type="button"
	>
		{@render fileIcon()}

		<div class="flex flex-col items-start gap-0.5">
			<span
				class="max-w-24 truncate text-sm font-medium text-foreground {readonly
					? ''
					: 'group-hover:pr-6'} md:max-w-32"
			>
				{name}
			</span>

			{@render info(pdfProcessingMode || (size ? formatFileSize(size) : undefined))}
		</div>

		{#if !readonly}
			{@render removeButton()}
		{/if}
	</button>
{/if}
