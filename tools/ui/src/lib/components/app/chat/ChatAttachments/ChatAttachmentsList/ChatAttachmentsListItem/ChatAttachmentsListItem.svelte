<script lang="ts">
	import {
		ChatAttachmentsListItemMcpPrompt,
		ChatAttachmentsListItemMcpResource,
		ChatAttachmentsListItemThumbnailFile,
		ChatAttachmentsListItemThumbnailImage
	} from '$lib/components/app';
	import { AttachmentType } from '$lib/enums';
	import type {
		ChatAttachmentDisplayItem,
		DatabaseMessageExtraMcpPrompt,
		DatabaseMessageExtraMcpResource,
		MCPResourceAttachment
	} from '$lib/types';
	import { isMcpPrompt, isMcpResource, isPdfFile } from '$lib/utils';

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
				name: extra.name,
				serverName: extra.serverName,
				title: extra.name,
				uri: extra.uri
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
						arguments: item.uploadedFile.mcpPrompt.arguments,
						content: item.textContent ?? '',
						name: item.name,
						promptName: item.uploadedFile.mcpPrompt.promptName,
						serverName: item.uploadedFile.mcpPrompt.serverName,
						type: AttachmentType.MCP_PROMPT as const
					}
				: null}
	{#if mcpPrompt}
		<ChatAttachmentsListItemMcpPrompt
			class="max-w-[300px] min-w-[200px] flex-shrink-0 {className} {scrollClasses}"
			isLoading={item.isLoading}
			loadError={item.loadError}
			onRemove={onFileRemove ? () => onFileRemove(item.id) : undefined}
			prompt={mcpPrompt}
			{readonly}
		/>
	{/if}
{:else if isMcpResource(item)}
	{@const mcpResource = item.attachment as DatabaseMessageExtraMcpResource}

	<ChatAttachmentsListItemMcpResource
		attachment={toMcpResourceAttachment(mcpResource, item.id)}
		class="flex-shrink-0 {className} {scrollClasses}"
		onclick={() => onMcpResourcePreview?.(mcpResource)}
	/>
{:else if item.isImage && item.preview}
	<ChatAttachmentsListItemThumbnailImage
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		height={imageHeight}
		id={item.id}
		{imageClass}
		name={item.name}
		onRemove={onFileRemove}
		onclick={() => onPreview?.(item)}
		preview={item.preview}
		{readonly}
		width={imageWidth}
	/>
{:else if isPdfFile(item.attachment, item.uploadedFile)}
	<ChatAttachmentsListItemThumbnailFile
		attachment={item.attachment}
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		id={item.id}
		name={item.name}
		onRemove={onFileRemove}
		onclick={() => onPreview?.(item)}
		{readonly}
		size={item.size}
		textContent={item.textContent}
		uploadedFile={item.uploadedFile}
	/>
{:else}
	<ChatAttachmentsListItemThumbnailFile
		attachment={item.attachment}
		class="flex-shrink-0 cursor-pointer {className} {scrollClasses}"
		id={item.id}
		name={item.name}
		onRemove={onFileRemove}
		onclick={() => onPreview?.(item)}
		{readonly}
		size={item.size}
		textContent={item.textContent}
		uploadedFile={item.uploadedFile}
	/>
{/if}
