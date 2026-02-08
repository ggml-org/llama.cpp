<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';

	interface Props {
		instructions?: string;
		class?: string;
	}

	let { instructions, class: className }: Props = $props();

	let isExpanded = $state(false);
</script>

{#if instructions}
	<Collapsible.Root bind:open={isExpanded} class={className}>
		<Collapsible.Trigger
			class="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
		>
			{#if isExpanded}
				<ChevronDown class="h-3.5 w-3.5" />
			{:else}
				<ChevronRight class="h-3.5 w-3.5" />
			{/if}

			<span>Server instructions</span>
		</Collapsible.Trigger>

		<Collapsible.Content class="mt-2">
			<p class="rounded bg-muted/50 p-2 text-xs text-muted-foreground">
				{instructions}
			</p>
		</Collapsible.Content>
	</Collapsible.Root>
{/if}
