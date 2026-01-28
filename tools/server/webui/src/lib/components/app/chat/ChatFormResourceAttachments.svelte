<script lang="ts">
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import {
		mcpResourceAttachments,
		mcpHasResourceAttachments
	} from '$lib/stores/mcp-resources.svelte';
	import { ChatAttachmentMcpResource, HorizontalScrollCarousel } from '$lib/components/app';

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
			{#each attachments as attachment (attachment.id)}
				<ChatAttachmentMcpResource
					{attachment}
					onRemove={handleRemove}
					onClick={() => handleResourceClick(attachment.resource.uri)}
				/>
			{/each}
		</HorizontalScrollCarousel>
	</div>
{/if}
