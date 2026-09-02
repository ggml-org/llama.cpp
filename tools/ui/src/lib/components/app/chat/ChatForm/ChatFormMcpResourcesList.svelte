<script lang="ts">
	import { ChatAttachmentsListItemMcpResource, ScrollCarousel } from '$lib/components/app';
	import { ScrollCarouselVariant } from '$lib/enums';
	import { mcpStore } from '$lib/stores';

	interface Props {
		class?: string;
		onResourceClick?: (uri: string) => void;
	}

	let { class: className, onResourceClick }: Props = $props();

	const attachments = $derived(mcpStore.resources.attachments);
	const hasAttachments = $derived(mcpStore.resources.hasAttachments);

	function handleRemove(attachmentId: string) {
		mcpStore.removeResourceAttachment(attachmentId);
	}

	function handleResourceClick(uri: string) {
		onResourceClick?.(uri);
	}
</script>

{#if hasAttachments}
	<div class={className}>
		<ScrollCarousel gapSize="2" variant={ScrollCarouselVariant.CENTER}>
			{#each attachments as attachment, i (attachment.id)}
				<ChatAttachmentsListItemMcpResource
					{attachment}
					class={i === 0 ? 'ml-3' : ''}
					onRemove={handleRemove}
					onclick={() => handleResourceClick(attachment.resource.uri)}
				/>
			{/each}
		</ScrollCarousel>
	</div>
{/if}
