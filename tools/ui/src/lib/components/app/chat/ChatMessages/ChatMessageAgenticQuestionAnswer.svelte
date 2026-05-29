<script lang="ts">
	import { MessageSquare } from '@lucide/svelte';
	import type { AgenticQuestionAnswerPair } from '$lib/utils/agentic-question';

	interface Props {
		pairs: AgenticQuestionAnswerPair[];
	}

	let { pairs }: Props = $props();
</script>

{#if pairs.length > 0}
	<div class="rounded-lg border border-border bg-card p-3 shadow-sm">
		<div class="mb-2 flex min-w-0 items-center gap-2">
			<MessageSquare class="h-4 w-4 shrink-0 text-muted-foreground" />
			<div class="truncate text-sm font-medium">Question answered</div>
		</div>

		<div class="space-y-2">
			{#each pairs as pair, index (`${index}:${pair.question}`)}
				<div class="rounded-md bg-muted/30 px-2 py-1.5">
					<div class="text-sm leading-5 break-words">{pair.question}</div>
					<div class="mt-1 text-sm leading-5 break-words text-muted-foreground">
						{pair.answer || 'No answer'}
					</div>
				</div>
			{/each}
		</div>
	</div>
{/if}
