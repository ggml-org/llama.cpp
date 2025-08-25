<script lang="ts">
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import type { MessageSiblingInfo } from '$lib/utils/branching';

	interface Props {
		siblingInfo: MessageSiblingInfo | null;
		onNavigateToSibling?: (siblingId: string) => void;
		class?: string;
	}

	let {
		siblingInfo,
		onNavigateToSibling,
		class: className = ''
	}: Props = $props();

	let hasPrevious = $derived(siblingInfo && siblingInfo.currentIndex > 0);
	let hasNext = $derived(siblingInfo && siblingInfo.currentIndex < siblingInfo.totalSiblings - 1);
	let previousSiblingId = $derived(hasPrevious ? siblingInfo!.siblingIds[siblingInfo!.currentIndex - 1] : null);
	let nextSiblingId = $derived(hasNext ? siblingInfo!.siblingIds[siblingInfo!.currentIndex + 1] : null);

	function handlePrevious() {
		if (previousSiblingId) {
			onNavigateToSibling?.(previousSiblingId);
		}
	}

	function handleNext() {
		if (nextSiblingId) {
			onNavigateToSibling?.(nextSiblingId);
		}
	}
</script>

{#if siblingInfo && siblingInfo.totalSiblings > 1}
	<div 
		class="flex items-center gap-1 text-xs text-muted-foreground {className}"
		role="navigation"
		aria-label="Message version {siblingInfo.currentIndex + 1} of {siblingInfo.totalSiblings}"
	>
		<Tooltip>
			<TooltipTrigger>
				<Button
					variant="ghost"
					size="sm"
					class="h-5 w-5 p-0 {!hasPrevious ? 'opacity-30 cursor-not-allowed' : ''}"
					onclick={handlePrevious}
					disabled={!hasPrevious}
					aria-label="Previous message version"
				>
					<ChevronLeft class="h-3 w-3" />
				</Button>
			</TooltipTrigger>
			<TooltipContent>
				<p>Previous version</p>
			</TooltipContent>
		</Tooltip>

		<span class="px-1 font-mono text-xs">
			{siblingInfo.currentIndex + 1}/{siblingInfo.totalSiblings}
		</span>

		<Tooltip>
			<TooltipTrigger>
				<Button
					variant="ghost"
					size="sm"
					class="h-5 w-5 p-0 {!hasNext ? 'opacity-30 cursor-not-allowed' : ''}"
					onclick={handleNext}
					disabled={!hasNext}
					aria-label="Next message version"
				>
					<ChevronRight class="h-3 w-3" />
				</Button>
			</TooltipTrigger>
			<TooltipContent>
				<p>Next version</p>
			</TooltipContent>
		</Tooltip>
	</div>
{/if}
