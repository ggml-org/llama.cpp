<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Badge } from '$lib/components/ui/badge';
	import McpServerCardActions from './McpServerCardActions.svelte';

	interface Tool {
		name: string;
		description?: string;
	}

	interface Props {
		tools: Tool[];
		isHealthChecking: boolean;
		onEdit: () => void;
		onRefresh: () => void;
		onDelete: () => void;
	}

	let { tools, isHealthChecking, onEdit, onRefresh, onDelete }: Props = $props();

	let isExpanded = $state(false);
	let toolsCount = $derived(tools.length);
</script>

<Collapsible.Root bind:open={isExpanded}>
	<div class="flex items-center justify-between gap-3">
		<Collapsible.Trigger
			class="flex flex-1 items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
		>
			{#if isExpanded}
				<ChevronDown class="h-3.5 w-3.5" />
			{:else}
				<ChevronRight class="h-3.5 w-3.5" />
			{/if}
			<span>{toolsCount} tools available · Show details</span>
		</Collapsible.Trigger>
		<McpServerCardActions {isHealthChecking} {onEdit} {onRefresh} {onDelete} />
	</div>
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
