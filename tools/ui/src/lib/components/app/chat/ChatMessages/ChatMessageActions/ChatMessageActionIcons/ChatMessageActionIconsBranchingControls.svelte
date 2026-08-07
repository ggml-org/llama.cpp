<script lang="ts">
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';

	interface Props {
		class?: string;
		siblingInfo: ChatMessageSiblingInfo | null;
		onNavigateToSibling?: (siblingId: string) => void;
	}

	let { class: className = '', siblingInfo, onNavigateToSibling }: Props = $props();

	let hasPrevious = $derived(siblingInfo && siblingInfo.currentIndex > 0);
	let hasNext = $derived(siblingInfo && siblingInfo.currentIndex < siblingInfo.totalSiblings - 1);
	let nextSiblingId = $derived(
		hasNext ? siblingInfo!.siblingIds[siblingInfo!.currentIndex + 1] : null
	);
	let previousSiblingId = $derived(
		hasPrevious ? siblingInfo!.siblingIds[siblingInfo!.currentIndex - 1] : null
	);
</script>

{#if siblingInfo && siblingInfo.totalSiblings > 1}
	<div
		aria-label="Message version {siblingInfo.currentIndex + 1} of {siblingInfo.totalSiblings}"
		class="flex items-center gap-1 text-xs text-muted-foreground {className}"
		role="navigation"
	>
		<ActionIcon
			icon={ChevronLeft}
			tooltip="Previous version"
			disabled={!hasPrevious}
			class="h-5 w-5 p-0 {!hasPrevious ? '!cursor-not-allowed opacity-30' : ''}"
			onclick={() => onNavigateToSibling?.(previousSiblingId!)}
		/>

		<span class="px-1 font-mono text-xs">
			{siblingInfo.currentIndex + 1}/{siblingInfo.totalSiblings}
		</span>

		<ActionIcon
			icon={ChevronRight}
			tooltip="Next version"
			disabled={!hasNext}
			class="h-5 w-5 p-0 {!hasNext ? 'opacity-30' : ''}"
			onclick={() => onNavigateToSibling?.(nextSiblingId!)}
		/>
	</div>
{/if}
