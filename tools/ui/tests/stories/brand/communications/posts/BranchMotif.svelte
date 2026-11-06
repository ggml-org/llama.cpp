<!--
	A decision tree, for decision models, in 3D: from one question the options fan out in depth and branch again,
	each branch as heavy and as dark as it is likely, the chosen path in cinder with its probability in Mono, and the
	runner-up's beside it. A motif for posts about decision models, choices and probabilities.
-->
<script lang="ts">
	import './motifs/motif.css';
	import { far, type Point3, project, pts, type View } from './motifs/project';

	interface Props {
		/** the chosen option's probability and the runner-up's */
		values?: [string, string];
	}

	let { values = ['0.91', '0.06'] }: Props = $props();

	const view: View = { cx: 205, cy: 140, distance: 6, pitch: 0.42, scale: 96, yaw: -0.58 };
	const p = (q: Point3) => project(q, view);

	interface Node {
		at: Point3;
		children: Node[];
		/** the path's probability, from the root */
		p: number;
	}

	// at every step, three options, the first the likeliest; deeper levels fan out tighter
	const odds = [0.82, 0.12, 0.06];
	const first = [0.91, 0.06, 0.03];
	const fan: [number, number][] = [
		[0, -0.7],
		[1, 0.6],
		[-1, -0.3]
	];
	const xs = [-2, -0.7, 0.55, 1.75];
	const spreads = [1.05, 0.45, 0.18];
	const grow = (at: Point3, level: number, p: number): Node => ({
		at,
		children:
			level === 3
				? []
				: (level === 0 ? first : odds).map((q, i) =>
						grow(
							[
								xs[level + 1],
								at[1] + fan[i][0] * spreads[level],
								at[2] + fan[i][1] * spreads[level]
							],
							level + 1,
							p * q
						)
					),
		p
	});
	const tree = grow([xs[0], 0, 0], 0, 1);

	// a branch as a smooth S in 3D, sampled and projected
	const curve = (a: Point3, b: Point3) =>
		Array.from({ length: 17 }, (_, i) => {
			const t = i / 16;
			const s = t * t * (3 - 2 * t);

			return p([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
		});

	// the chosen path: the likeliest child at each step, which runs through the middle
	const chosen: Node[] = [];
	for (let n: Node | undefined = tree; n; n = n.children[0]) chosen.push(n);

	const all: Node[] = [];
	const walk = (n: Node) => {
		all.push(n);
		n.children.forEach(walk);
	};
	walk(tree);

	const branches = all.flatMap((n) =>
		n.children.map((c) => ({ chosen: chosen.includes(c), line: curve(n.at, c.at), p: c.p }))
	);
	const nodes = all
		.map((n) => ({ chosen: chosen.includes(n), p: n.p, pt: p(n.at) }))
		.sort((a, b) => b.pt[2] - a.pt[2]);

	const label = (n: Node, dx: number, dy: number) => {
		const [x, y] = p(n.at);

		return { x: x + dx, y: y + dy };
	};
	const top = label(tree.children[0], -14, 22);
	const runner = label(tree.children[1], 0, -14);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#each branches.filter((b) => !b.chosen) as b, i (i)}
		<polyline
			style:opacity={0.22 + 0.55 * Math.sqrt(b.p) * (1 - 0.4 * far(b.line[8][2], 3))}
			style:stroke-width={1.1 + 6 * b.p}
			class="branch"
			points={pts(b.line)}
		/>
	{/each}

	{#each branches.filter((b) => b.chosen) as b, i (i)}
		<polyline style:stroke-width="4" class="accent-line" points={pts(b.line)} />
	{/each}

	{#each nodes as n, i (i)}
		<circle
			style:opacity={n.chosen ? 1 : 0.4 + 0.5 * (1 - far(n.pt[2], 3))}
			class={n.chosen ? 'accent' : 'solid'}
			cx={n.pt[0]}
			cy={n.pt[1]}
			r={n.chosen ? 6.5 : 2.6 + 3.4 * Math.sqrt(n.p)}
		/>
	{/each}

	<text class="chosen" text-anchor="end" x={top.x} y={top.y}>{values[0]}</text>

	<text text-anchor="middle" x={runner.x} y={runner.y}>{values[1]}</text>
</svg>

<style>
	.branch {
		fill: none;
		stroke: var(--p-ink);
		stroke-linecap: round;
	}
	text.chosen {
		fill: var(--art-cinder);
	}
</style>
