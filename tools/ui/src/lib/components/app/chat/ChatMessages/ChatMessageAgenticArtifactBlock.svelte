<script lang="ts">
	import { Download, FileText, Image as ImageIcon } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { AttachmentType } from '$lib/enums';
	import { createBase64DataUrl, formatFileSize, getFileTypeLabel } from '$lib/utils';
	import type { DatabaseMessageExtra } from '$lib/types';

	interface Props {
		attachment: DatabaseMessageExtra;
		artifactKey: string;
	}

	let { attachment, artifactKey }: Props = $props();

	function getMimeType(item: DatabaseMessageExtra): string {
		if ('mimeType' in item && item.mimeType) return item.mimeType;
		if (item.type === AttachmentType.TEXT) return 'text/plain';
		if (item.type === AttachmentType.PDF) return 'application/pdf';
		if (item.type === AttachmentType.IMAGE) return 'image';
		return String(item.type).toLowerCase();
	}

	function getImageUrl(item: DatabaseMessageExtra): string | null {
		if (item.type === AttachmentType.IMAGE && 'base64Url' in item) return item.base64Url;
		return null;
	}

	function getBinaryUrl(item: DatabaseMessageExtra): string | null {
		if (item.type === AttachmentType.IMAGE && 'base64Url' in item) return item.base64Url;
		if (item.type === AttachmentType.TEXT && 'content' in item) {
			const mimeType = getMimeType(item);
			return `data:${mimeType};charset=utf-8,${encodeURIComponent(item.content)}`;
		}
		if (item.type === AttachmentType.PDF && 'base64Data' in item) {
			return createBase64DataUrl('application/pdf', item.base64Data);
		}
		if (
			(item.type === AttachmentType.AUDIO || item.type === AttachmentType.VIDEO) &&
			'base64Data' in item &&
			'mimeType' in item
		) {
			return createBase64DataUrl(item.mimeType, item.base64Data);
		}
		return null;
	}

	function openArtifact() {
		window.dispatchEvent(
			new CustomEvent('agentic-artifact-open', {
				detail: {
					key: artifactKey,
					attachment
				}
			})
		);
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			openArtifact();
		}
	}

	const mimeType = $derived(getMimeType(attachment));
	const imageUrl = $derived(getImageUrl(attachment));
	const binaryUrl = $derived(getBinaryUrl(attachment));
	const size = $derived(
		'size' in attachment && attachment.size ? formatFileSize(attachment.size) : ''
	);
</script>

<div
	class="artifact-header artifact-chat-block"
	role="button"
	tabindex="0"
	onclick={openArtifact}
	onkeydown={handleKeydown}
>
	<div class="flex min-w-0 items-center gap-3">
		<div
			class="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary"
		>
			{#if imageUrl}
				<ImageIcon class="h-4 w-4" />
			{:else}
				<FileText class="h-4 w-4" />
			{/if}
		</div>

		<div class="min-w-0">
			<div class="truncate text-sm font-medium">{attachment.name}</div>
			<div class="mt-0.5 text-xs text-muted-foreground">
				{getFileTypeLabel(mimeType)}{size ? ` · ${size}` : ''}
			</div>
		</div>
	</div>

	<Button
		aria-label="Download artifact"
		class="h-8 shrink-0 px-2"
		disabled={!binaryUrl}
		href={binaryUrl ?? undefined}
		download={attachment.name}
		size="sm"
		variant="outline"
		onclick={(event) => event.stopPropagation()}
	>
		<Download class="h-4 w-4" />
		Download
	</Button>
</div>

<style>
	.artifact-chat-block {
		display: flex;
		min-width: 0;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		border: 1px solid hsl(var(--border));
		border-radius: 0.5rem;
		background: hsl(var(--card));
		padding: 0.75rem 1rem;
		box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
		cursor: pointer;
		transition:
			background-color 150ms ease,
			border-color 150ms ease;
	}

	.artifact-chat-block:hover {
		background: hsl(var(--accent) / 0.35);
	}

	.artifact-chat-block:focus-visible {
		outline: 2px solid hsl(var(--ring));
		outline-offset: 2px;
	}
</style>
