<script lang="ts">
	import { PROCESSING_INFO_TIMEOUT } from '$lib/constants/processing-info';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { slotsService } from '$lib/services/slots';
	import { isLoading, activeMessages, activeConversation } from '$lib/stores/chat.svelte';
	import { config } from '$lib/stores/settings.svelte';

	const processingState = useProcessingState();

	let processingDetails = $derived(processingState.getProcessingDetails());

	let showSlotsInfo = $derived(isLoading() || config().keepStatsVisible);

	$effect(() => {
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible || isLoading()) {
			processingState.startMonitoring();
		}

		if (!isLoading() && !keepStatsVisible) {
			setTimeout(() => {
				if (!config().keepStatsVisible) {
					processingState.stopMonitoring();
				}
			}, PROCESSING_INFO_TIMEOUT);
		}
	});

	$effect(() => {
		activeConversation();

		const messages = activeMessages() as DatabaseMessage[];
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible) {
			if (messages.length === 0) {
				slotsService.clearState();
				return;
			}

			let foundTimingData = false;

			for (let i = messages.length - 1; i >= 0; i--) {
				const message = messages[i];
				if (message.role === 'assistant' && message.timings) {
					foundTimingData = true;

					slotsService
						.updateFromTimingData({
							prompt_n: message.timings.prompt_n || 0,
							predicted_n: message.timings.predicted_n || 0,
							predicted_per_second:
								message.timings.predicted_n && message.timings.predicted_ms
									? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
									: 0,
							cache_n: message.timings.cache_n || 0
						})
						.catch((error) => {
							console.warn('Failed to update processing state from stored timings:', error);
						});
					break;
				}
			}

			if (!foundTimingData) {
				slotsService.clearState();
			}
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
		pointer-events: none;
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.chat-processing-info-container.visible {
		opacity: 1;
		pointer-events: auto;
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
