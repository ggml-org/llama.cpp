<script lang="ts">
	import { Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	let enabled = $derived(conversationsStore.getThinkingEnabled());
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<span class="h-3 w-3 shrink-0 cursor-default rounded">
			{#if enabled}
				<Lightbulb class="h-3 w-3 fill-amber-400 text-amber-400" />
			{:else}
				<LightbulbOff class="h-3 w-3 fill-muted-foreground text-muted-foreground" />
			{/if}
		</span>
	</Tooltip.Trigger>
	<Tooltip.Content>
		<p>Thinking {enabled ? 'on' : 'off'}</p>
	</Tooltip.Content>
</Tooltip.Root>
