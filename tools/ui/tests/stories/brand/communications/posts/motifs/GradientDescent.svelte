<!--
	Gradient descent: a loss surface in 3D, drawn as a wire mesh that fades with distance, and the steps a model takes
	down it, from a ridge to the lowest valley, the last one in cinder. A motif for posts about training and
	fine-tuning.
-->
<script lang="ts">
	import './motif.css';
	import { far, project, pts, type View } from './project';

	const view: View = { cx: 200, cy: 150, distance: 4, pitch: 0.55, scale: 150, yaw: 0.62 };
	// a bowl with ripples, so the way down is not a straight line
	const loss = (x: number, z: number) =>
		0.4 * (0.8 * x * x + z * z) + 0.17 * Math.sin(3.2 * x) * Math.cos(2.6 * z) - 0.32;
	const at = (x: number, z: number) => project([x, loss(x, z), z], view);

	const n = 20;
	const samples = 32;
	const line = (fixed: number, across: boolean) =>
		Array.from({ length: samples + 1 }, (_, i) => {
			const t = -1 + (2 * i) / samples;

			return across ? at(t, fixed) : at(fixed, t);
		});
	const mesh = Array.from({ length: n + 1 }, (_, i) => -1 + (2 * i) / n).flatMap((f) => [
		line(f, true),
		line(f, false)
	]);
	const fade = (l: number[][]) =>
		0.12 + 0.5 * (1 - far(l.reduce((s, p) => s + p[2], 0) / l.length));

	// the descent: steps against the gradient from a high corner
	const steps = (() => {
		// start from the highest corner, so the way down is long
		let [x, z] = (
			[
				[0.95, 0.95],
				[0.95, -0.95],
				[-0.95, 0.95],
				[-0.95, -0.95]
			] as const
		).reduce((a, b) => (loss(b[0], b[1]) > loss(a[0], a[1]) ? b : a));

		const out: number[][] = [];
		const e = 0.001;

		for (let i = 0; i < 12; i++) {
			const p = project([x, loss(x, z) + 0.03, z], view);

			out.push(p);

			const gx = (loss(x + e, z) - loss(x - e, z)) / (2 * e);
			const gz = (loss(x, z + e) - loss(x, z - e)) / (2 * e);

			x -= 0.2 * gx;
			z -= 0.2 * gz;
		}

		return out;
	})();
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each mesh as l, i (i)}
		<polyline style:opacity={fade(l)} class="mesh" points={pts(l)} />
	{/each}

	<polyline style:stroke-width="4" class="ink" points={pts(steps)} />

	{#each steps as [cx, cy], i (i)}
		{@const last = i === steps.length - 1}
		<circle class={last ? 'accent' : 'solid'} {cx} {cy} r={last ? 10 : 5.5} />
	{/each}
</svg>

<style>
	.mesh {
		fill: none;
		stroke: var(--p-ink);
		stroke-width: 1.4;
	}
</style>
