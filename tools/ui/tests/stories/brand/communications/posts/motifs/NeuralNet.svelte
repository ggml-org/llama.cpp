<!--
	A neural network: layers of nodes joined by faint weights, and one path through it in cinder, from an input to the
	output it lights up. A motif for posts about models and what is inside them.
-->
<script lang="ts">
	import './motif.css';

	const layers = [3, 5, 5, 2];
	const x = (l: number) => 40 + l * 106;
	const y = (n: number, count: number) => 150 + (n - (count - 1) / 2) * 56;
	// the lit path: a node in each layer
	const path = [1, 3, 1, 0];
	const nodes = layers.map((count, l) =>
		Array.from({ length: count }, (_, n) => [x(l), y(n, count)])
	);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each nodes.slice(0, -1) as layer, l (l)}
		{#each layer as a, i (i)}
			{#each nodes[l + 1] as b, j (j)}<line
					class="thin"
					x1={a[0]}
					x2={b[0]}
					y1={a[1]}
					y2={b[1]}
				/>{/each}
		{/each}
	{/each}

	<polyline class="accent-line" points={path.map((n, l) => nodes[l][n].join(',')).join(' ')} />

	{#each nodes as layer, l (l)}
		{#each layer as [cx, cy], n (n)}
			<circle class={path[l] === n ? 'accent' : 'solid'} {cx} {cy} r="13" />
		{/each}
	{/each}
</svg>
