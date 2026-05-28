<script lang="ts">
	import { Files } from '@lucide/svelte';
	import { ChatAttachmentsList } from '$lib/components/app';
	import { dedupeArtifactAttachments, getArtifactAttachmentKey } from '$lib/utils';
	import ChatMessageAgenticArtifactBlock from './ChatMessageAgenticArtifactBlock.svelte';
	import type { DatabaseMessageExtra } from '$lib/types';

	interface Props {
		attachments?: DatabaseMessageExtra[];
		messageId?: string;
	}

	let { attachments = [], messageId = '' }: Props = $props();

	const artifactAttachments = $derived(dedupeArtifactAttachments(attachments));
	const fileAttachments = $derived(
		attachments.filter((attachment) => attachment.presentation !== 'artifact')
	);

	function getArtifactKey(attachment: DatabaseMessageExtra, index: number): string {
		return getArtifactAttachmentKey(messageId, attachment, index);
	}
</script>

{#if artifactAttachments.length > 0 || fileAttachments.length > 0}
	<div class="grid gap-3">
		{#each artifactAttachments as attachment, index (getArtifactKey(attachment, index))}
			<ChatMessageAgenticArtifactBlock
				{attachment}
				artifactKey={getArtifactKey(attachment, index)}
			/>
		{/each}

		{#if fileAttachments.length > 0}
			<div class="rounded-lg border border-border bg-card p-3 shadow-sm">
				<div class="mb-2 flex min-w-0 items-center gap-2">
					<Files class="h-4 w-4 shrink-0 text-muted-foreground" />
					<div class="truncate text-sm font-medium">
						File{fileAttachments.length === 1 ? '' : 's'}
					</div>
					<div class="shrink-0 text-xs text-muted-foreground">{fileAttachments.length}</div>
				</div>

				<ChatAttachmentsList
					attachments={fileAttachments}
					readonly
					imageHeight="h-28"
					imageWidth="w-auto"
				/>
			</div>
		{/if}
	</div>
{/if}
