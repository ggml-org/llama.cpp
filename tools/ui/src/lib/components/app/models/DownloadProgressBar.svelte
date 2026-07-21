<script lang="ts">
	interface Props {
		/** Bytes downloaded so far. Caller supplies the value; we normalize 0..1. */
		downloadedBytes: number;
		/** Total bytes for the download plan. */
		totalBytes: number;
		/** Pin to the bottom edge as a thin overlay like `ModelLoadHighlight`. */
		overlay?: boolean;
	}

	let { downloadedBytes, totalBytes, overlay = false }: Props = $props();

	let fraction = $derived.by(() => {
		if (totalBytes <= 0) return 0;
		return Math.min(Math.max(downloadedBytes / totalBytes, 0), 1);
	});
	let percent = $derived(Math.round(fraction * 100));
</script>

{#if overlay}
	<div class="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 overflow-hidden rounded-b-sm">
		<div
			class="h-full animate-pulse bg-primary transition-[width] duration-200 ease-out"
			style="width: {percent}%"
		></div>
	</div>
{:else}
	<div class="h-1 w-full overflow-hidden rounded-full bg-muted">
		<div
			class="h-full animate-pulse bg-primary transition-[width] duration-200 ease-out"
			style="width: {percent}%"
		></div>
	</div>
{/if}
