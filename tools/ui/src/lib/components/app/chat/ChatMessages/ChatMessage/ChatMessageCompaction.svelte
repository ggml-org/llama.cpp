<script lang="ts">
	import { FoldVertical } from '@lucide/svelte';
	import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
	import type { DatabaseMessage } from '$lib/types';

	interface Props {
		class?: string;
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	const COMPACTION_HEADER = 'Conversation compacted';
	const COMPACTION_HEADER_PENDING = 'Compacting conversation...';
	const COMPACTION_SUBTITLE = 'Later turns continue from this summary';

	// The node streams its summary in as it is generated; an empty content
	// therefore means the summary request is still in flight.
	const isPending = $derived(message.content.trim().length === 0);

	let open = $state(false);
</script>

<div class="border-border/50 rounded-lg border px-3 {className}">
	<CollapsibleContentBlock
		bind:open
		icon={FoldVertical}
		title={isPending ? COMPACTION_HEADER_PENDING : COMPACTION_HEADER}
		subtitle={isPending ? undefined : COMPACTION_SUBTITLE}
		shimmerTitle={isPending}
	>
		{#if !isPending}
			<div class="text-muted-foreground pb-2 text-sm">
				<MarkdownContent content={message.content} />
			</div>
		{/if}
	</CollapsibleContentBlock>
</div>
