<!--
	A neural network in 3D: its layers as square grids of nodes standing one behind another, each in a faint frame,
	every node joined to the next layer by a hairline that fades with depth, and one path through them in cinder.
	A motif for posts about models and what is inside them.
-->
<script lang="ts">
	import './motif.css';
	import { far, type Point3, project, pts, type View } from './project';

	const view: View = { cx: 200, cy: 150, distance: 6, pitch: 0.3, scale: 108, yaw: -0.32 };
	const p = (q: Point3) => project(q, view);

	const sizes = [4, 4, 4, 2];
	const xs = [-1.65, -0.55, 0.55, 1.65];
	const span = 0.85;
	const layers = sizes.map((n, l) =>
		Array.from({ length: n * n }, (_, i) => {
			const r = Math.floor(i / n);
			const c = i % n;
			const t = (k: number) =>
				(n === 1 ? 0 : -span + (2 * span * k) / (n - 1)) * (n === 2 ? 0.45 : 1);

			return p([xs[l], t(r), t(c)]);
		})
	);
	// each layer's frame, a little larger than its nodes
	const frames = xs.map((x, l) => {
		const e = (sizes[l] === 2 ? span * 0.45 : span) + 0.22;

		return (
			[
				[x, -e, -e],
				[x, e, -e],
				[x, e, e],
				[x, -e, e]
			] as Point3[]
		).map(p);
	});
	const edges = layers
		.slice(0, -1)
		.flatMap((layer, l) => layer.flatMap((a) => layers[l + 1].map((b) => [a, b])));
	// the lit path: a node in each layer
	const path = [5, 10, 6, 1].map((i, l) => layers[l][i]);
	const nodes = layers.flat().sort((a, b) => b[2] - a[2]);
	const lit = new Set(path);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each frames as f, i (i)}
		<polygon style:opacity={0.25 + 0.3 * (1 - far(f[0][2], 3))} class="frame" points={pts(f)} />
	{/each}

	{#each edges as [a, b], i (i)}
		<line
			style:opacity={0.05 + 0.14 * (1 - far(a[2] + b[2], 4))}
			class="edge"
			x1={a[0]}
			x2={b[0]}
			y1={a[1]}
			y2={b[1]}
		/>
	{/each}

	<polyline style:stroke-width="3.5" class="accent-line" points={pts(path)} />

	{#each nodes as n, i (i)}
		<circle
			style:opacity={lit.has(n) ? 1 : 0.4 + 0.6 * (1 - far(n[2], 3))}
			class={lit.has(n) ? 'accent' : 'solid'}
			cx={n[0]}
			cy={n[1]}
			r={(lit.has(n) ? 6.5 : 4) + 2 * (1 - far(n[2], 3))}
		/>
	{/each}
</svg>

<style>
	.frame {
		fill: var(--p-inset);
		stroke: var(--p-ink);
		stroke-width: 1.2;
		stroke-opacity: 0.35;
	}
	.edge {
		stroke: var(--p-ink);
		stroke-width: 1;
	}
</style>
