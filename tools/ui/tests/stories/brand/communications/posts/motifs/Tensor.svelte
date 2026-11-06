<!--
	A tensor: a 3 × 3 × 3 block of cubes with small gaps between them, each face shaded by which way it faces (the
	tops lightest), one cube in cinder, and its shape in Mono. A motif for posts about weights, layers and quantization.
-->
<script lang="ts">
	import './motif.css';
	import { type Point3, project, pts, type View } from './project';

	const view: View = { cx: 200, cy: 126, distance: 14, pitch: 0.52, scale: 98, yaw: -0.62 };
	const p = (q: Point3) => project(q, view);

	const n = 3;
	const size = 0.5;
	const step = 0.62;
	const h = size / 2;

	// the six faces as a normal and its four corners, by unit offsets
	const faces: { normal: Point3; corners: Point3[]; shade: number }[] = [
		{
			corners: [
				[-1, 1, -1],
				[1, 1, -1],
				[1, 1, 1],
				[-1, 1, 1]
			],
			normal: [0, 1, 0],
			shade: 0.04
		},
		{
			corners: [
				[-1, -1, -1],
				[1, -1, -1],
				[1, -1, 1],
				[-1, -1, 1]
			],
			normal: [0, -1, 0],
			shade: 0.7
		},
		{
			corners: [
				[1, -1, -1],
				[1, 1, -1],
				[1, 1, 1],
				[1, -1, 1]
			],
			normal: [1, 0, 0],
			shade: 0.2
		},
		{
			corners: [
				[-1, -1, -1],
				[-1, 1, -1],
				[-1, 1, 1],
				[-1, -1, 1]
			],
			normal: [-1, 0, 0],
			shade: 0.2
		},
		{
			corners: [
				[-1, -1, 1],
				[1, -1, 1],
				[1, 1, 1],
				[-1, 1, 1]
			],
			normal: [0, 0, 1],
			shade: 0.38
		},
		{
			corners: [
				[-1, -1, -1],
				[1, -1, -1],
				[1, 1, -1],
				[-1, 1, -1]
			],
			normal: [0, 0, -1],
			shade: 0.38
		}
	];

	const cubes = Array.from({ length: n ** 3 }, (_, i) => {
		const c: Point3 = [
			(Math.floor(i / 9) - 1) * step,
			((Math.floor(i / 3) % 3) - 1) * step,
			((i % 3) - 1) * step
		];
		const depth = p(c)[2];
		const visible = faces
			.filter(
				(f) =>
					p([c[0] + f.normal[0] * h, c[1] + f.normal[1] * h, c[2] + f.normal[2] * h])[2] < depth
			)
			.map((f) => ({
				points: pts(f.corners.map(([x, y, z]) => p([c[0] + x * h, c[1] + y * h, c[2] + z * h]))),
				shade: f.shade
			}));

		return { c, depth, visible };
	}).sort((a, b) => b.depth - a.depth);

	// the cinder cube: the top one nearest the viewer
	const accent = cubes.filter((q) => q.c[1] > 0).at(-1);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each cubes as cube, i (i)}
		{#each cube.visible as face, j (j)}
			<polygon class={cube === accent ? 'accent' : 'base'} points={face.points} />

			<polygon
				style:fill-opacity={cube === accent ? face.shade * 0.5 : face.shade}
				class="shade"
				points={face.points}
			/>
		{/each}
	{/each}

	<text text-anchor="middle" x="200" y="294">[3, 3, 3]</text>
</svg>

<style>
	.base {
		fill: var(--p-base);
	}
	.shade {
		fill: var(--p-ink);
		stroke: var(--p-ink);
		stroke-width: 1.2;
		stroke-linejoin: round;
	}
</style>
