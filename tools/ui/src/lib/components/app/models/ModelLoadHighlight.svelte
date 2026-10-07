<script lang="ts">
	let { percent }: { percent: number } = $props();
</script>

<!-- thin determinate load bar pinned to the bottom edge, pulsing while it fills -->
<div class="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 overflow-hidden rounded-b-sm">
	<div
		class="h-full animate-pulse bg-primary transition-[width] duration-200 ease-out"
		style="width: {percent}%"
	></div>
</div>
