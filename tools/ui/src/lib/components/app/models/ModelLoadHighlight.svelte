<script lang="ts">
	let { percent }: { percent: number } = $props();
</script>

<!-- determinate load fill that highlights the pill left to right, pulsing on the whole surface -->
<div
	class="pointer-events-none absolute inset-y-0 left-0 animate-pulse rounded-sm bg-primary/25 transition-[width] duration-200 ease-out"
	style="width: {percent}%"
></div>
