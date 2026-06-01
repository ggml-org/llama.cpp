<script lang="ts">
	import { Lightbulb } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { DatabaseMessage } from '$lib/types/database';

	interface Props {
		message: DatabaseMessage;
	}

	let { message }: Props = $props();

	// For assistant messages, show the bulb if there's reasoning content.
	// No chat template detection needed — the model already produced reasoning,
	// so it obviously supports it. Template detection is only for the chat form
	// selector where we don't have a message to inspect.
	let hasReasoning = $derived(!!message.reasoningContent);
</script>

{#if hasReasoning}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<span class="h-3 w-3 shrink-0 cursor-default rounded">
				<Lightbulb class="h-3 w-3 fill-amber-400 text-amber-400" />
			</span>
		</Tooltip.Trigger>
		<Tooltip.Content>
			<p>Thinking on</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}
