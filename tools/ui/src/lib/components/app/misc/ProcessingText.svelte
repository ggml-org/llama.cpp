<script lang="ts">
	import { fade } from 'svelte/transition';
	let {
		processingText,
		cls
	}: {
		processingText: string;
		cls: string;
	} = $props();
</script>

<div class="{cls} w-full max-w-[48rem]" in:fade>
	<div class="processing-container">
		<span class="processing-text">
			{processingText}
		</span>
	</div>
</div>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}
</style>
