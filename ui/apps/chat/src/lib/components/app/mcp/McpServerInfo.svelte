<script lang="ts">
	import * as Collapsible from '@ggml-org/llama-ui/ui/collapsible';
	import { ChevronDown, ChevronRight } from '@lucide/svelte';

	interface Props {
		instructions?: string;
		class?: string;
	}

	let { class: className, instructions }: Props = $props();

	let isExpanded = $state(false);
</script>

{#if instructions}
	<Collapsible.Root bind:open={isExpanded} class={className}>
		<Collapsible.Trigger
			class="flex w-full items-center gap-1 text-caption text-muted-foreground hover:text-foreground"
		>
			{#if isExpanded}
				<ChevronDown class="size-icon-sm" />
			{:else}
				<ChevronRight class="size-icon-sm" />
			{/if}

			<span>Server instructions</span>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<p class="mt-2 rounded bg-muted/50 p-2 text-caption text-muted-foreground">
				{instructions}
			</p>
		</Collapsible.Content>
	</Collapsible.Root>
{/if}
