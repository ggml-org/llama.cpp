<script lang="ts">
	import { Brain } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		reasoningContent: string | null;
		isStreaming?: boolean;
		hasRegularContent?: boolean;
		class?: string;
	}

	let { reasoningContent, isStreaming = false, hasRegularContent = false, class: className = '' }: Props = $props();

	let isExpanded = $state(true);

	// Auto-collapse when regular content starts (even while streaming)
	$effect(() => {
		if (hasRegularContent && reasoningContent) {
			isExpanded = false;
		}
	});
</script>

<Collapsible.Root bind:open={isExpanded} class="mb-6 {className}">
	<Card class="border-muted bg-muted/30 gap-0 py-0">
		<Collapsible.Trigger class="cursor-pointer flex items-center justify-between p-3">
			<div class="text-muted-foreground flex items-center gap-2">
				<Brain class="h-4 w-4" />
				<span class="text-sm font-medium">
					{isStreaming ? 'Reasoning...' : 'Reasoning'}
				</span>
			</div>
			
			<div
				class={buttonVariants({ 
					variant: "ghost", 
					size: "sm", 
					class: "h-6 w-6 p-0 text-muted-foreground hover:text-foreground" 
				})}
			>
				<ChevronsUpDownIcon class="h-4 w-4" />
				<span class="sr-only">Toggle reasoning content</span>
			</div>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div class="border-muted border-t px-3 pb-3">
				<div class="pt-3">
					<MarkdownContent content={reasoningContent || ''} class="text-xs leading-relaxed" />
				</div>
			</div>
		</Collapsible.Content>
	</Card>
</Collapsible.Root>
