<!--
	How the icon sits in a circle crop (the avatars, the PWA icon): the same square twice, the icon set at the exact
	centre and as the files place it, lifted a little, with its box, both centre lines and the margins above and below
	drawn in, and a circle in the gap between the crop and each of the icon's two rounded corners (the blade's top left,
	the foot's bottom right), touching both. The placement comes from the file itself (its translate and scale), so the
	figure follows the files.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../logo-showcase/lib/logo';

	interface Props {
		/** the canvas's side and the icon's placement in it, as in its SVG */
		placement: { scale: number; size: number; x: number; y: number };
	}

	let { placement }: Props = $props();

	const side = $derived(SIZE * placement.scale);
	const centred = $derived((placement.size - side) / 2);
	// the lift as a fraction of the icon's height: 1/150 in the brand's files
	const lift = $derived(Math.round(side / (centred - placement.y)));

	type Point = [number, number];

	/** A path's rounded corner, in the icon's own 600 units: its curve's four control points. */
	function roundedCorner(d: string): Point[] {
		let at: Point = [0, 0];

		for (const [, command, args] of d.matchAll(/([MLCHVZ])([^MLCHVZ]*)/g)) {
			const n = args.trim()
				? args
						.trim()
						.split(/[\s,]+/)
						.map(Number)
				: [];

			if (command === 'C') return [at, [n[0], n[1]], [n[2], n[3]], [n[4], n[5]]];

			if (command === 'H') at = [n[0], at[1]];
			else if (command === 'V') at = [at[0], n[0]];
			else if (command !== 'Z') at = [n[0], n[1]];
		}

		throw new Error('a path without a rounded corner');
	}

	// the foot's rounded bottom right and the blade's rounded top left, the two outer corners that round off
	const CORNERS = PATHS.map(roundedCorner);
	const CORNER_NAMES = ['foot', 'blade'];

	/**
	 * The gap between each rounded corner and the circle crop, with the icon at top y: the circle that touches the
	 * corner at its apex (the curve's midpoint, along the corner's bisector) and the crop, and its diameter.
	 */
	function gaps(y: number) {
		const c = placement.size / 2;

		return CORNERS.map(([p0, p1, p2, p3], i) => {
			// the apex and the outward normal there, at t = 1/2
			const apex = [0, 1].map((k) => (p0[k] + 3 * p1[k] + 3 * p2[k] + p3[k]) / 8);
			const tangent = [0, 1].map((k) => p3[k] + p2[k] - p1[k] - p0[k]);
			const length = Math.hypot(tangent[0], tangent[1]);

			let n = [tangent[1] / length, -tangent[0] / length];

			if (n[0] * (apex[0] - SIZE / 2) + n[1] * (apex[1] - SIZE / 2) < 0) n = [-n[0], -n[1]];

			const v = [placement.x + apex[0] * placement.scale - c, y + apex[1] * placement.scale - c];
			// |v + r n| = c - r, solved for r
			const r = (c * c - v[0] ** 2 - v[1] ** 2) / (2 * (v[0] * n[0] + v[1] * n[1] + c));

			return {
				cx: c + v[0] + r * n[0],
				cy: c + v[1] + r * n[1],
				gap: 2 * r,
				name: CORNER_NAMES[i]
			};
		});
	}

	/** Which rounded corner comes nearer the crop, or that they're even (within half a pixel). */
	function balance(g: ReturnType<typeof gaps>) {
		const [a, b] = g;

		if (Math.abs(a.gap - b.gap) < 0.5) return 'the rounded corners even';

		return `the ${a.gap < b.gap ? a.name : b.name}’s corner nearer the circle`;
	}

	const panels = $derived(
		[
			{ label: 'At the exact centre', y: centred },
			{ label: `Lifted 1/${lift} of its height, as in the files`, y: placement.y }
		].map((panel) => {
			const g = gaps(panel.y);

			return { ...panel, gaps: g, note: balance(g) };
		})
	);

	const round = (n: number) => Math.round(n * 10) / 10;
</script>

<figure class="m-0 grid min-w-0 gap-3">
	<div class="grid grid-cols-2 gap-4">
		{#each panels as panel (panel.label)}
			<div class="grid min-w-0 gap-2">
				<svg
					class="block aspect-square w-full overflow-hidden rounded-card border border-border/60"
					viewBox="0 0 {placement.size} {placement.size}"
				>
					<rect fill="#FAFAFA" height={placement.size} width={placement.size} />

					<!-- the circle it is cropped to -->
					<circle
						class="guide muted"
						cx={placement.size / 2}
						cy={placement.size / 2}
						r={placement.size / 2}
					/>

					<!-- the square's centre line, then the icon's own -->
					<line
						class="guide muted"
						x1="0"
						x2={placement.size}
						y1={placement.size / 2}
						y2={placement.size / 2}
					/>

					<g fill="#111111" transform="translate({placement.x} {panel.y}) scale({placement.scale})">
						{#each PATHS as d (d)}<path {d} />{/each}
					</g>

					<rect class="guide" height={side} width={side} x={placement.x} y={panel.y} />

					<line
						class="guide solid"
						x1={placement.x}
						x2={placement.x + side}
						y1={panel.y + side / 2}
						y2={panel.y + side / 2}
					/>

					<!-- the gaps where the icon comes nearest the circle crop, each filled by a circle touching both -->
					{#each panel.gaps as g (g.name)}
						<circle class="gap" cx={g.cx} cy={g.cy} r={g.gap / 2} />

						<text class="margin" font-size={placement.size * 0.04} x={g.cx} y={g.cy}
							>{round(g.gap)}</text
						>
					{/each}

					<!-- the margins above and below the icon's box, set aside from the corners' circles -->
					<text
						class="margin"
						font-size={placement.size * 0.045}
						x={placement.size * 0.65}
						y={panel.y / 2}>{round(panel.y)}</text
					>

					<text
						class="margin"
						font-size={placement.size * 0.045}
						x={placement.size * 0.35}
						y={(placement.size + panel.y + side) / 2}>{round(placement.size - panel.y - side)}</text
					>
				</svg>

				<span class="grid text-center">
					<span class="text-sm">{panel.label}</span>

					<span class="text-xs text-muted-foreground">{panel.note}</span>
				</span>
			</div>
		{/each}
	</div>

	<figcaption class="text-xs text-muted-foreground">
		The icon fills {Math.round((side / placement.size) * 100)}% of the side. The circles fill the
		gaps between the circle crop and the icon’s two rounded corners, the blade’s top left and the
		foot’s bottom right: {panels
			.map((p) => p.gaps.map((g) => round(g.gap)).join(' and '))
			.join(' px at the exact centre, ')}
		px lifted 1/{lift} ({round(centred - placement.y)} px at {placement.size} × {placement.size}).
	</figcaption>
</figure>

<style>
	.guide {
		fill: none;
		stroke: #ee6c1e;
		stroke-dasharray: 4 3;
		stroke-width: 1px;
		vector-effect: non-scaling-stroke;
	}
	.guide.solid {
		stroke-dasharray: none;
	}
	.guide.muted {
		stroke: #a3a3a3;
	}
	.gap {
		fill: color-mix(in srgb, #ee6c1e 12%, transparent);
		stroke: #ee6c1e;
		stroke-width: 1px;
		vector-effect: non-scaling-stroke;
	}
	.margin {
		dominant-baseline: middle;
		fill: #ee6c1e;
		font-family: var(--font-mono, monospace);
		text-anchor: middle;
	}
</style>
