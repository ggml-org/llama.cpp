<script lang="ts">
	import { Eye } from '@lucide/svelte';
	import { AttachmentType } from '$lib/enums';
	import { ATTACHMENT_SAVED_REGEX } from '$lib/constants/agentic';
	import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile, DatabaseMessageExtraAudioFile } from '$lib/types';
	import { type AgenticSection } from '$lib/utils';
	import { parseReadMediaMeta } from './parsers/read-media';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const readMediaMeta = $derived(parseReadMediaMeta(section));

	// Find the attachment from toolResultExtras (attached to the tool result message.
	// The extractBase64Attachments function in agentic.svelte.ts replaces the data URI line
	// with [Attachment saved: name] and stores the base64 as an extra.
	const mediaAttachment = $derived.by(() => {
		const extras = section.toolResultExtras;
		if (!extras || extras.length === 0) return null;
		// Extract the attachment name from the cleaned result text
		const match = section.toolResult?.match(ATTACHMENT_SAVED_REGEX);
		if (!match) return null;
		const attachmentName = match[1];
		return extras.find(
			(e): e is DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile =>
				(e.type === AttachmentType.IMAGE || e.type === AttachmentType.AUDIO) &&
				e.name === attachmentName
		) ?? null;
	});

	const isAudio = $derived(mediaAttachment?.type === AttachmentType.AUDIO);
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={readMediaMeta} {onToggle}>
	{#snippet titleSnippet()}
		<span class="text-muted-foreground">Read media </span>
		<span class="font-mono">{readMediaMeta?.fileName}</span>
	{/snippet}

	{#snippet children(_meta, _ctx)}
		{#if section.toolResult}
			{#if mediaAttachment}
				{#if isAudio}
					<div class="mt-2">
						<audio controls class="w-full rounded-lg">
							<source src={mediaAttachment.base64Url} type={readMediaMeta?.mimeType ?? 'audio/mpeg'} />
							Your browser does not support the audio element.
						</audio>
					</div>
				{:else}
					<div class="mt-2">
						<img
							src={mediaAttachment.base64Url}
							alt={readMediaMeta?.fileName ?? 'media'}
							class="max-h-[60vh] max-w-full rounded-lg object-contain shadow-lg"
							loading="lazy"
						/>
					</div>
				{/if}
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Media attachment not found in message extras
				</div>
			{/if}

			{#if readMediaMeta?.sizeBytes || readMediaMeta?.mimeType}
				<div class="mt-2 flex gap-4 text-xs text-muted-foreground">
					{#if readMediaMeta?.sizeBytes}
						<span>Size: {readMediaMeta.sizeBytes} bytes</span>
					{/if}
					{#if readMediaMeta?.mimeType}
						<span>MIME: {readMediaMeta.mimeType}</span>
					{/if}
				</div>
			{/if}

			{#if readMediaMeta?.path}
				<div class="mt-1 text-xs text-muted-foreground/60 font-mono">{readMediaMeta.path}</div>
			{/if}
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for media data...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
