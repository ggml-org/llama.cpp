<script lang="ts">
	let {
		visible,
		processingDetails
	}: {
		visible: boolean;
		processingDetails: string[];
	} = $props();
</script>

<div class="processing-info-container pointer-events-none" class:visible>
	<div class="processing-info-content">
		{#each processingDetails as detail (detail)}
			<span class="processing-info-detail pointer-events-auto backdrop-blur-sm">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.processing-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		padding: 0 1rem 0.75rem;
		opacity: 0;
		transform: translateY(50%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.processing-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.processing-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.processing-info-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		border-radius: 0.375rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
		white-space: nowrap;
	}

	@media (max-width: 768px) {
		.processing-info-content {
			gap: 0.5rem;
		}

		.processing-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
