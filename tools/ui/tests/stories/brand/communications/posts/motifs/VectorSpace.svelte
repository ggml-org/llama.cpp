<!--
	A vector space in 3D: a floor grid that recedes, three axes, clusters of points that sit together by meaning, and
	two vectors from the origin, one in cinder with a line down to the floor to place it. Points shrink and fade with
	depth. A motif for posts about embeddings, search and retrieval.
-->
<script lang="ts">
	import './motif.css';
	import { far, type Point3, project, random, type View } from './project';

	const view: View = { cx: 200, cy: 185, distance: 5, pitch: 0.5, scale: 150, yaw: 0.55 };
	const p = (q: Point3) => project(q, view);
	const floor = 0;

	const ticks = Array.from({ length: 9 }, (_, i) => -1 + i * 0.25);
	const grid = ticks.flatMap((t) => [
		[p([t, floor, -1]), p([t, floor, 1])],
		[p([-1, floor, t]), p([1, floor, t])]
	]);

	const centres: Point3[] = [
		[0.62, 0.55, -0.45],
		[-0.6, 0.35, 0.4],
		[0.45, 0.2, 0.7]
	];
	const rnd = random(11);
	const cloud = centres
		.flatMap((c) =>
			Array.from({ length: 14 }, () =>
				p([c[0] + (rnd() - 0.5) * 0.5, c[1] + (rnd() - 0.5) * 0.5, c[2] + (rnd() - 0.5) * 0.5])
			)
		)
		.sort((a, b) => b[2] - a[2]);

	const o = p([0, floor, 0]);
	const axes: Point3[] = [
		[1.1, floor, 0],
		[0, 1.05, 0],
		[0, floor, 1.1]
	];
	const a = p(centres[0]);
	const b = p(centres[1]);
	const drop = p([centres[0][0], floor, centres[0][2]]);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each grid as [s, e], i (i)}
		<line
			style:opacity={0.1 + 0.18 * (1 - far(s[2] + e[2]))}
			class="thin"
			x1={s[0]}
			x2={e[0]}
			y1={s[1]}
			y2={e[1]}
		/>
	{/each}

	{#each axes as axis (axis.join())}
		{@const e = p(axis)}
		<line
			style:opacity="0.45"
			style:stroke-width="2.5"
			class="ink"
			x1={o[0]}
			x2={e[0]}
			y1={o[1]}
			y2={e[1]}
		/>
	{/each}

	{#each cloud as [cx, cy, d], i (i)}
		<circle
			style:opacity={0.25 + 0.5 * (1 - far(d))}
			class="solid"
			{cx}
			{cy}
			r={3 + 3 * (1 - far(d))}
		/>
	{/each}

	<line
		style:opacity="0.7"
		style:stroke-dasharray="4 5"
		class="thin"
		x1={a[0]}
		x2={drop[0]}
		y1={a[1]}
		y2={drop[1]}
	/>

	<line style:stroke-width="5" class="ink" x1={o[0]} x2={b[0]} y1={o[1]} y2={b[1]} />

	<circle class="solid" cx={b[0]} cy={b[1]} r="8" />

	<line style:stroke-width="5" class="accent-line" x1={o[0]} x2={a[0]} y1={o[1]} y2={a[1]} />

	<circle class="accent" cx={a[0]} cy={a[1]} r="9" />

	<circle class="solid" cx={o[0]} cy={o[1]} r="4" />
</svg>
