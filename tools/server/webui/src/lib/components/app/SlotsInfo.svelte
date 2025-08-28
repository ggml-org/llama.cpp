<script lang="ts">
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';

	const processingState = useProcessingState();

	let showSlotsInfo = $derived(isLoading());

	let processingDetails = $derived(processingState.getProcessingDetails());

	// Monitor during loading and add delay before stopping to capture final updates
	$effect(() => {
		if (isLoading()) {
			processingState.startMonitoring();
		} else {
			// Delay stopping to capture final context updates after streaming
			setTimeout(() => {
				processingState.stopMonitoring();
			}, 2000); // 2 second delay to ensure we get final updates
		}
	});
</script>

<div class="slots-info-container" class:visible={showSlotsInfo}>
	<div class="slots-info-content">
		{#each processingDetails as detail (detail)}
			<span class="slots-info-detail">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.slots-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		background: var(--background);
		backdrop-filter: blur(8px);
		background: rgba(var(--background-rgb), 0.95);
		padding: 0.75rem 1rem;
		margin-bottom: 1rem;
		opacity: 0;
		transform: translateY(-100%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.slots-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.slots-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.slots-info-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		background: var(--muted);
		border-radius: 0.375rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
		white-space: nowrap;
	}

	@media (max-width: 768px) {
		.slots-info-content {
			gap: 0.5rem;
		}

		.slots-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
