<script lang="ts">
	import { AttachmentType } from '$lib/enums';
	import type {
		ChatAttachmentDisplayItem,
		DatabaseMessageExtraMcpPrompt,
		DatabaseMessageExtraMcpResource,
		MCPResourceAttachment
	} from '$lib/types';
	import { isPdfFile } from '$lib/utils';
	import { isMcpPrompt, isMcpResource } from '$lib/utils/attachment-display';
	import {
		ChatAttachmentsItemMcpPrompt,
		ChatAttachmentsItemMcpResource,
		ChatAttachmentsItemThumbnailImage,
		ChatAttachmentsItemThumbnailFile
	} from '$lib/components/app';

	interface Props {
		class?: string;
		imageClass?: string;
		imageHeight?: string;
		imageWidth?: string;
		item: ChatAttachmentDisplayItem;
		limitToSingleRow?: boolean;
		onFileRemove?: (fileId: string) => void;
		onMcpResourcePreview?: (extra: DatabaseMessageExtraMcpResource) => void;
		onPreview?: (item: ChatAttachmentDisplayItem) => void;
		readonly?: boolean;
	}

	let {
		class: className = '',
		imageClass = '',
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		item,
		limitToSingleRow = false,
		onFileRemove,
		onMcpResourcePreview,
		onPreview,
		readonly = false
	}: Props = $props();

	const scrollClasses = $derived(limitToSingleRow ? 'first:ml-4 last:mr-4' : '');

	function toMcpResourceAttachment(
		extra: DatabaseMessageExtraMcpResource,
		id: string
	): MCPResourceAttachment {
		return {
			id,
			resource: {
				uri: extra.uri,
				name: extra.name,
				title: extra.name,
				serverName: extra.serverName
			}
		};
	}
</script>

{#if isMcpPrompt(item)}
	{@const mcpPrompt =
		item.attachment?.type === AttachmentType.MCP_PROMPT
			? (item.attachment as DatabaseMessageExtraMcpPrompt)
			: item.uploadedFile?.mcpPrompt
				? {
						type: AttachmentType.MCP_PROMPT as const,
						name: item.name,
						serverName: item.uploadedFile.mcpPrompt.serverName,
						promptName: item.uploadedFile.mcpPrompt.promptName,
						content: item.textContent ?? '',
						arguments: item.uploadedFile.mcpPrompt.arguments
					}
				: null}
	{#if mcpPrompt}
		<ChatAttachmentsItemMcpPrompt
			class="max-w-[300px] min-w-[200px] flex-shrink-0 {className} {scrollClasses}"
			prompt={mcpPrompt}
			{readonly}
			isLoading={item.isLoading}
			loadError={item.loadError}
			onRemove={onFileRemove ? () => onFileRemove(item.id) : undefined}
		/>
	{/if}
{:else if isMcpResource(item)}
	{@const mcpResource = item.attachment as DatabaseMessageExtraMcpResource}

	<ChatAttachmentsItemMcpResource
		class="flex-shrink-0 {className} {scrollClasses}"
		attachment={toMcpResourceAttachment(mcpResource, item.id)}
		onclick={() => onMcpResourcePreview?.(mcpResource)}
	/>
{:else if item.isImage && item.preview}
	<ChatAttachmentsItemThumbnailImage
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		id={item.id}
		name={item.name}
		preview={item.preview}
		{readonly}
		onRemove={onFileRemove}
		height={imageHeight}
		width={imageWidth}
		{imageClass}
		onclick={() => onPreview?.(item)}
	/>
{:else if isPdfFile(item.attachment, item.uploadedFile)}
	<ChatAttachmentsItemThumbnailFile
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		id={item.id}
		name={item.name}
		size={item.size}
		{readonly}
		onRemove={onFileRemove}
		textContent={item.textContent}
		attachment={item.attachment}
		uploadedFile={item.uploadedFile}
		onclick={() => onPreview?.(item)}
	/>
{:else}
	<ChatAttachmentsItemThumbnailFile
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		id={item.id}
		name={item.name}
		size={item.size}
		{readonly}
		onRemove={onFileRemove}
		textContent={item.textContent}
		attachment={item.attachment}
		uploadedFile={item.uploadedFile}
		onclick={() => onPreview?.(item)}
	/>
{/if}
