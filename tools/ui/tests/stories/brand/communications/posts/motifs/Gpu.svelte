<!--
	A GPU: a graphics card with two fans, its heatsink fins and the edge connector, and its memory in Mono, the
	activity light in cinder. A motif for posts about hardware, offload and speed.
-->
<script lang="ts">
	import './motif.css';

	const fans = [130, 270];
	const blades = Array.from({ length: 7 }, (_, i) => (i * 360) / 7);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<rect class="ink" height="150" rx="16" width="340" x="30" y="50" />

	{#each fans as cx (cx)}
		<circle class="ink" {cx} cy="125" r="54" />

		{#each blades as angle (angle)}
			<path
				style:stroke-width="4"
				class="ink"
				d="M {cx} 125 q 14 -24 0 -44"
				transform="rotate({angle} {cx} 125)"
			/>
		{/each}

		<circle class="solid" {cx} cy="125" r="9" />
	{/each}

	<path style:stroke-width="5" class="ink" d="M 70 200 v 26 h 120 v -26" />

	{#each Array(9) as _, i (i)}<line
			style:stroke-width="4"
			class="ink"
			x1={84 + i * 12}
			x2={84 + i * 12}
			y1="208"
			y2="220"
		/>{/each}

	<circle class="accent" cx="346" cy="72" r="8" />

	<text text-anchor="end" x="370" y="250">24 GB</text>
</svg>
