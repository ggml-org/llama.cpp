<!--
	How the icon sits in a circle crop (the avatars, the PWA icon), the same square three times: the icon at the exact
	centre, lifted 1/60 of its height (the first optical centre), and as the files place it, lifted and nudged left. Its
	box, the centre lines and the margins all round are drawn in, and a circle fills the gap between the crop and each of
	the icon's four outer corners, touching both: the two rounded ones (the blade's top left, the foot's bottom right)
	and the two sharp ones (the blade's top right, the foot's bottom left). The files' placement comes from the file
	itself (its translate and scale), so the figure follows the files.
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
	// the files' lift and nudge, as fractions of the icon's size: 1/39 up and 1/31 left in the brand's files
	const lift = $derived(Math.round(side / (centred - placement.y)));
	const nudge = $derived(Math.round(side / (centred - placement.x)));

	type Point = [number, number];

	/**
	 * A path's two outer corners, in the icon's own 600 units: its rounded corner (its curve's control points) and its
	 * sharp one (its corner farthest from the icon's centre).
	 */
	function outerCorners(d: string) {
		const points: Point[] = [];

		let curve: Point[] = [];
		let at: Point = [0, 0];

		for (const [, command, args] of d.matchAll(/([MLCHVZ])([^MLCHVZ]*)/g)) {
			const n = args.trim()
				? args
						.trim()
						.split(/[\s,]+/)
						.map(Number)
				: [];

			if (command === 'C') {
				curve = [at, [n[0], n[1]], [n[2], n[3]], [n[4], n[5]]];
				at = [n[4], n[5]];
			} else if (command === 'H') at = [n[0], at[1]];
			else if (command === 'V') at = [at[0], n[0]];
			else if (command !== 'Z') at = [n[0], n[1]];

			points.push(at);
		}

		const reach = ([x, y]: Point) => Math.hypot(x - SIZE / 2, y - SIZE / 2);

		return { curve, sharp: points.reduce((far, p) => (reach(p) > reach(far) ? p : far)) };
	}

	// the foot, then the blade
	const CORNERS = PATHS.map(outerCorners);
	const NAMES = ['foot', 'blade'];

	/**
	 * The gaps between the circle crop and the icon's outer corners, with the icon at x y: each filled by the circle
	 * that touches the crop and the corner, at a rounded corner's apex (its curve's midpoint, along its bisector) and
	 * at a sharp corner's point, and measured by that circle's diameter.
	 */
	function gaps(x: number, y: number) {
		const c = placement.size / 2;
		const onCanvas = ([px, py]: number[]) => [
			x + px * placement.scale - c,
			y + py * placement.scale - c
		];
		// the circle touching the crop and the point v (from the centre), along n: |v + r n| = c - r, solved for r
		const fill = (v: number[], n: number[]) => {
			const r = (c * c - v[0] ** 2 - v[1] ** 2) / (2 * (v[0] * n[0] + v[1] * n[1] + c));

			return { cx: c + v[0] + r * n[0], cy: c + v[1] + r * n[1], gap: 2 * r };
		};

		return CORNERS.flatMap(({ curve: [p0, p1, p2, p3], sharp }, i) => {
			const apex = [0, 1].map((k) => (p0[k] + 3 * p1[k] + 3 * p2[k] + p3[k]) / 8);
			const tangent = [0, 1].map((k) => p3[k] + p2[k] - p1[k] - p0[k]);
			const length = Math.hypot(tangent[0], tangent[1]);

			let n = [tangent[1] / length, -tangent[0] / length];

			if (n[0] * (apex[0] - SIZE / 2) + n[1] * (apex[1] - SIZE / 2) < 0) n = [-n[0], -n[1]];

			const v = onCanvas(sharp);
			const d = Math.hypot(v[0], v[1]);

			return [
				{ ...fill(onCanvas(apex), n), kind: 'rounded', name: NAMES[i] },
				{ ...fill(v, [v[0] / d, v[1] / d]), kind: 'sharp', name: NAMES[i] }
			];
		});
	}

	/** For each kind of corner, whether the pair is even (within half a pixel) or which comes nearer the crop. */
	function balance(g: ReturnType<typeof gaps>) {
		return ['rounded', 'sharp']
			.map((kind) => {
				const [a, b] = g.filter((gap) => gap.kind === kind);

				if (Math.abs(a.gap - b.gap) < 0.5) return `${kind} corners even`;

				return `${kind}: the ${a.gap < b.gap ? a.name : b.name}’s nearer`;
			})
			.join(' · ');
	}

	const panels = $derived(
		[
			{ label: 'At the exact centre', short: 'Exact centre', x: centred, y: centred },
			{
				label: 'Lifted 1/60, the first optical centre',
				short: 'Lifted 1/60',
				x: centred,
				y: centred - side / 60
			},
			{
				label: `Lifted 1/${lift} and nudged 1/${nudge} left, as in the files`,
				short: `Lifted 1/${lift}, nudged 1/${nudge} left`,
				x: placement.x,
				y: placement.y
			}
		].map((panel) => {
			const g = gaps(panel.x, panel.y);

			return { ...panel, gaps: g, note: balance(g) };
		})
	);

	const round = (n: number) => Math.round(n * 10) / 10;
</script>

<figure class="m-0 grid min-w-0 gap-3">
	<!-- the three placements as they show, light and dark in the circle crop, with nothing drawn in -->
	<div class="grid grid-cols-3 gap-4">
		{#each panels as panel (panel.label)}
			<div class="grid min-w-0 content-start gap-2">
				<div class="grid grid-cols-2 gap-2">
					{#each [['#FAFAFA', '#111111'], ['#111111', '#FAFAFA']] as [ground, ink] (ground)}
						<svg
							class="block aspect-square w-full rounded-full border border-border/60"
							viewBox="0 0 {placement.size} {placement.size}"
						>
							<circle
								cx={placement.size / 2}
								cy={placement.size / 2}
								fill={ground}
								r={placement.size / 2}
							/>

							<g fill={ink} transform="translate({panel.x} {panel.y}) scale({placement.scale})">
								{#each PATHS as d (d)}<path {d} />{/each}
							</g>
						</svg>
					{/each}
				</div>

				<span class="text-center text-xs">{panel.short}</span>
			</div>
		{/each}
	</div>

	<div class="mt-4 grid gap-4 sm:grid-cols-3">
		{#each panels as panel (panel.label)}
			<div class="grid min-w-0 content-start gap-2">
				<svg
					class="block aspect-square w-full overflow-hidden rounded-card border border-border/60"
					viewBox="0 0 {placement.size} {placement.size}"
				>
					<rect fill="#FAFAFA" height={placement.size} width={placement.size} />

					<!-- the circle it is cropped to, and its centre lines -->
					<circle
						class="guide muted"
						cx={placement.size / 2}
						cy={placement.size / 2}
						r={placement.size / 2}
					/>

					<line
						class="guide muted"
						x1="0"
						x2={placement.size}
						y1={placement.size / 2}
						y2={placement.size / 2}
					/>

					<line
						class="guide muted"
						x1={placement.size / 2}
						x2={placement.size / 2}
						y1="0"
						y2={placement.size}
					/>

					<g fill="#111111" transform="translate({panel.x} {panel.y}) scale({placement.scale})">
						{#each PATHS as d (d)}<path {d} />{/each}
					</g>

					<!-- the icon's box and its centre lines -->
					<rect class="guide" height={side} width={side} x={panel.x} y={panel.y} />

					<line
						class="guide solid"
						x1={panel.x}
						x2={panel.x + side}
						y1={panel.y + side / 2}
						y2={panel.y + side / 2}
					/>

					<line
						class="guide solid"
						x1={panel.x + side / 2}
						x2={panel.x + side / 2}
						y1={panel.y}
						y2={panel.y + side}
					/>

					<!-- the gaps between the crop and the icon's outer corners, each filled by a circle touching both -->
					{#each panel.gaps as g (g.kind + g.name)}
						<circle class={['gap', g.kind]} cx={g.cx} cy={g.cy} r={g.gap / 2} />

						<text class="margin" font-size={placement.size * 0.04} x={g.cx} y={g.cy}
							>{round(g.gap)}</text
						>
					{/each}

					<!-- the margins round the icon's box: above and below in the square's corners, outside the crop -->
					{#each [{ at: [0.08, panel.y / 2], value: panel.y }, { at: [0.92, (placement.size + panel.y + side) / 2], value: placement.size - panel.y - side }] as m (m.at[0])}
						<text
							class="margin"
							font-size={placement.size * 0.045}
							x={placement.size * m.at[0]}
							y={m.at[1]}>{round(m.value)}</text
						>
					{/each}

					{#each [{ value: panel.x, x: panel.x / 2 }, { value: placement.size - panel.x - side, x: (placement.size + panel.x + side) / 2 }] as m (m.x)}
						<text
							class="margin"
							font-size={placement.size * 0.045}
							x={m.x}
							y={placement.size * 0.42}>{round(m.value)}</text
						>
					{/each}
				</svg>

				<span class="grid text-center">
					<span class="text-sm">{panel.label}</span>

					<span class="text-xs text-muted-foreground">{panel.note}</span>
				</span>
			</div>
		{/each}
	</div>

	<figcaption class="text-xs text-muted-foreground">
		The icon fills {Math.round((side / placement.size) * 100)}% of the side; the numbers are pixels
		at
		{placement.size} × {placement.size}. Each circle fills the gap between the circle crop and one
		of the icon’s outer corners: the rounded ones, the blade’s top left and the foot’s bottom right,
		and the sharp ones, the blade’s top right and the foot’s bottom left. Centred, the foot sits
		nearer the crop; the first lift, 1/60, evens the weight but not the corners; lifted 1/{lift} and nudged
		1/{nudge} left, each pair of corners is as far from the crop as the other, so the circle clears the
		icon evenly all round.
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
	.gap.sharp {
		fill: color-mix(in srgb, #a3a3a3 14%, transparent);
		stroke: #a3a3a3;
	}
	.margin {
		dominant-baseline: middle;
		fill: #ee6c1e;
		font-family: var(--font-mono, monospace);
		text-anchor: middle;
	}
</style>
