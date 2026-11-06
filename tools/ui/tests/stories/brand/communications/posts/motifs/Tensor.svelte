<!--
	A tensor: three 4 × 4 slices stacked in depth, the back ones fainter, one value in cinder, and its shape in Mono.
	A motif for posts about weights, layers and quantization.
-->
<script lang="ts">
	import './motif.css';

	const n = 4;
	const cell = 40;
	const slices = [2, 1, 0];
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each slices as s (s)}
		<g style:opacity={1 - s * 0.32} transform="translate({110 + s * 34} {50 - s * 26})">
			<rect class="soft" height={n * cell} width={n * cell} />

			{#each Array(n + 1) as _, i (i)}
				<line style:stroke-width="4" class="ink" x1="0" x2={n * cell} y1={i * cell} y2={i * cell} />

				<line style:stroke-width="4" class="ink" x1={i * cell} x2={i * cell} y1="0" y2={n * cell} />
			{/each}

			{#if s === 0}<rect
					class="accent"
					height={cell - 12}
					width={cell - 12}
					x={cell * 2 + 6}
					y={cell + 6}
				/>{/if}
		</g>
	{/each}

	<text text-anchor="middle" x="190" y="290">[3, 4, 4]</text>
</svg>
