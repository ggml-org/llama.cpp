<script lang="ts">
	import {
		ChatAttachmentsListItemMcpResource,
		HorizontalScrollCarousel
	} from '$lib/components/app';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import {
		mcpHasResourceAttachments,
		mcpResourceAttachments
	} from '$lib/stores/mcp-resources.svelte';

	interface Props {
		class?: string;
		onResourceClick?: (uri: string) => void;
	}

	let { class: className, onResourceClick }: Props = $props();

	const attachments = $derived(mcpResourceAttachments());
	const hasAttachments = $derived(mcpHasResourceAttachments());

	function handleRemove(attachmentId: string) {
		mcpStore.removeResourceAttachment(attachmentId);
	}

	function handleResourceClick(uri: string) {
		onResourceClick?.(uri);
	}
</script>

{#if hasAttachments}
	<div class={className}>
		<HorizontalScrollCarousel gapSize="2">
			{#each attachments as attachment, i (attachment.id)}
				<ChatAttachmentsListItemMcpResource
					class={i === 0 ? 'ml-3' : ''}
					{attachment}
					onRemove={handleRemove}
					onclick={() => handleResourceClick(attachment.resource.uri)}
				/>
			{/each}
		</HorizontalScrollCarousel>
	</div>
{/if}
