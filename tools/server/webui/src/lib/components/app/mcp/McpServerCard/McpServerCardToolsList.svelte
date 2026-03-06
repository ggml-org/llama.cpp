<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Badge } from '$lib/components/ui/badge';

	interface Tool {
		name: string;
		description?: string;
	}

	interface Props {
		tools: Tool[];
	}

	let { tools }: Props = $props();

	let isExpanded = $state(false);
	let toolsCount = $derived(tools.length);
</script>

<Collapsible.Root bind:open={isExpanded}>
	<Collapsible.Trigger
		class="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
	>
		{#if isExpanded}
			<ChevronDown class="h-3.5 w-3.5" />
		{:else}
			<ChevronRight class="h-3.5 w-3.5" />
		{/if}

		<span>{toolsCount} tools available · Show details</span>
	</Collapsible.Trigger>

	<Collapsible.Content class="mt-2">
		<div class="max-h-64 space-y-3 overflow-y-auto">
			{#each tools as tool (tool.name)}
				<div>
					<Badge variant="secondary">{tool.name}</Badge>

					{#if tool.description}
						<p class="mt-1 text-xs text-muted-foreground">{tool.description}</p>
					{/if}
				</div>
			{/each}
		</div>
	</Collapsible.Content>
</Collapsible.Root>
