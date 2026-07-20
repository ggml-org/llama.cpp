<script lang="ts">
	import { Eye } from '@lucide/svelte';
	import { AttachmentType } from '$lib/enums';
	import { ATTACHMENT_SAVED_REGEX } from '$lib/constants/agentic';
	import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types';
	import { type AgenticSection } from '$lib/utils';
	import { parseReadImageMeta } from './parsers/read-image';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const readImageMeta = $derived(parseReadImageMeta(section));

	// Find the image attachment from toolResultExtras (attached to the tool result message.
	// The extractBase64Attachments function in agentic.svelte.ts replaces the data URI line
	// with [Attachment saved: name] and stores the base64 as an extra.
	const imageAttachment = $derived.by(() => {
		const extras = section.toolResultExtras;
		if (!extras || extras.length === 0) return null;
		// Extract the attachment name from the cleaned result text
		const match = section.toolResult?.match(ATTACHMENT_SAVED_REGEX);
		if (!match) return null;
		const attachmentName = match[1];
		return extras.find(
			(e): e is DatabaseMessageExtraImageFile =>
				e.type === AttachmentType.IMAGE && e.name === attachmentName
		) ?? null;
	});
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={readImageMeta} {onToggle}>
	{#snippet titleSnippet()}
		<span class="text-muted-foreground">Read image </span>
		<span class="font-mono">{readImageMeta?.fileName}</span>
	{/snippet}

	{#snippet children(_meta, _ctx)}
		{#if section.toolResult}
			{#if imageAttachment}
				<div class="mt-2">
					<img
						src={imageAttachment.base64Url}
						alt={readImageMeta?.fileName ?? 'image'}
						class="max-h-[60vh] max-w-full rounded-lg object-contain shadow-lg"
						loading="lazy"
					/>
				</div>
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Image attachment not found in message extras
				</div>
			{/if}

			{#if readImageMeta?.sizeBytes || readImageMeta?.mimeType}
				<div class="mt-2 flex gap-4 text-xs text-muted-foreground">
					{#if readImageMeta?.sizeBytes}
						<span>Size: {readImageMeta.sizeBytes} bytes</span>
					{/if}
					{#if readImageMeta?.mimeType}
						<span>MIME: {readImageMeta.mimeType}</span>
					{/if}
				</div>
			{/if}

			{#if readImageMeta?.path}
				<div class="mt-1 text-xs text-muted-foreground/60 font-mono">{readImageMeta.path}</div>
			{/if}
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for image data...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
