<script lang="ts">
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';

	const processingState = useProcessingState();

	let processingDetails = $derived(processingState.getProcessingDetails());

	let showSlotsInfo = $derived(isLoading());

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

<div class="chat-processing-info-container" class:visible={showSlotsInfo}>
	<div class="chat-processing-info-content">
		{#each processingDetails as detail (detail)}
			<span class="chat-processing-info-detail">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.chat-processing-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		padding: 1.5rem 1rem;
		opacity: 0;
		transform: translateY(50%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.chat-processing-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.chat-processing-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.chat-processing-info-detail {
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
		.chat-processing-info-content {
			gap: 0.5rem;
		}

		.chat-processing-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
