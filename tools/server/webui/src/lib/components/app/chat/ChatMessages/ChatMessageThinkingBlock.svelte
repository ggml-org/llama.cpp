<script lang="ts">
	import { Brain } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import { config } from '$lib/stores/settings.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		hasRegularContent?: boolean;
		isStreaming?: boolean;
		reasoningContent: string | null;
		children?: Snippet;
	}

	let {
		class: className = '',
		hasRegularContent = false,
		isStreaming = false,
		reasoningContent,
		children
	}: Props = $props();

	const currentConfig = config();

	// Expand automatically only while streaming when "Show thought in progress" is enabled.
	const initialAutoExpand = isStreaming && currentConfig.showThoughtInProgress;
	let isExpanded = $state(initialAutoExpand);
	let autoExpanded = $state(initialAutoExpand);

	$effect(() => {
		if (isStreaming && currentConfig.showThoughtInProgress && !isExpanded && !autoExpanded) {
			isExpanded = true;
			autoExpanded = true;
		} else if (!isStreaming && autoExpanded) {
			// Only collapse if this session auto-opened it; user manual toggles stay respected.
			isExpanded = false;
			autoExpanded = false;
		}
	});
</script>

<Collapsible.Root bind:open={isExpanded} class="{hasRegularContent ? 'mb-4' : 'mb-6'} {className}">
	<Card class="gap-0 border-muted bg-muted/30 py-0">
		<Collapsible.Trigger
			class="flex cursor-pointer items-center justify-between p-3"
			onclick={() => {
				autoExpanded = false; // user choice overrides auto behavior
			}}
		>
			<div class="flex items-center gap-2 text-muted-foreground">
				<Brain class="h-4 w-4" />

				<span class="text-sm font-medium">
					{isStreaming ? 'Reasoning...' : 'Reasoning'}
				</span>
			</div>

			<div
				class={buttonVariants({
					variant: 'ghost',
					size: 'sm',
					class: 'h-6 w-6 p-0 text-muted-foreground hover:text-foreground'
				})}
			>
				<ChevronsUpDownIcon class="h-4 w-4" />

				<span class="sr-only">Toggle reasoning content</span>
			</div>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div class="border-t border-muted px-3 pb-3">
				<div class="pt-3">
					<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
						{#if children}
							{@render children()}
						{:else}
							{reasoningContent ?? ''}
						{/if}
					</div>
				</div>
			</div>
		</Collapsible.Content>
	</Card>
</Collapsible.Root>
