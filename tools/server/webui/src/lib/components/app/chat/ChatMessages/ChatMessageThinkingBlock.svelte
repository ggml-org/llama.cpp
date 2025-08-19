<script lang="ts">
	import { ChevronDown, Brain } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { MarkdownContent } from '$lib/components/app';
	import { slide } from 'svelte/transition';

	interface Props {
		reasoningContent: string | null;
		isStreaming?: boolean;
		class?: string;
	}

	let { reasoningContent, isStreaming = false, class: className = '' }: Props = $props();

	let isExpanded = $state(false);
</script>

<Card class="border-muted bg-muted/30 mb-6 gap-0 py-0 {className}">
	<Button
		variant="ghost"
		class="h-auto w-full justify-between p-3 font-normal"
		onclick={() => (isExpanded = !isExpanded)}
	>
		<div class="text-muted-foreground flex items-center gap-2">
			<Brain class="h-4 w-4" />

			<span class="text-sm">
				{isStreaming ? 'Reasoning...' : 'Reasoning'}
			</span>
		</div>

		<ChevronDown
			class="text-muted-foreground h-4 w-4 transition-transform duration-200 {isExpanded
				? 'rotate-180'
				: ''}"
		/>
	</Button>

	{#if isExpanded}
		<div class="border-muted border-t px-3 pb-3" transition:slide={{ duration: 200 }}>
			<div class="pt-3">
				<MarkdownContent content={reasoningContent || ''} class="text-xs leading-relaxed" />
			</div>
		</div>
	{/if}
</Card>
