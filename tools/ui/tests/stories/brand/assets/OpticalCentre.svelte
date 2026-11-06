<!--
	How the icon sits in the avatars: the same square twice, the icon set at the exact centre and as the files place
	it, lifted a little, with its box, both centre lines and the margins above and below drawn in, and a circle in each
	of the two gaps where the icon comes nearest the circle crop, touching both. At the exact centre the foot comes
	nearer the crop than the blade; lifted, the two gaps are even. The placement comes from the avatar file itself (its
	translate and scale), so the figure follows the files.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../logo-showcase/lib/logo';

	interface Props {
		/** the avatar's side and the icon's placement in it, as in its SVG */
		placement: { scale: number; size: number; x: number; y: number };
	}

	let { placement }: Props = $props();

	const side = $derived(SIZE * placement.scale);
	const centred = $derived((placement.size - side) / 2);
	// the lift as a fraction of the icon's height: 1/150 in the brand's files
	const lift = $derived(Math.round(side / (centred - placement.y)));

	type Point = [number, number];

	/** The icon's outline as points, in its own 600 units: its paths' corners, and their curves sampled. */
	function outline(d: string): Point[] {
		const points: Point[] = [];

		for (const [, command, args] of d.matchAll(/([MLCHVZ])([^MLCHVZ]*)/g)) {
			const n = args.trim()
				? args
						.trim()
						.split(/[\s,]+/)
						.map(Number)
				: [];
			const [x0, y0] = points.at(-1) ?? [0, 0];

			if (command === 'H') points.push([n[0], y0]);
			else if (command === 'V') points.push([x0, n[0]]);
			else if (command === 'C')
				for (let i = 1; i <= 24; i++) {
					const t = i / 24;
					const u = 1 - t;
					const b = [u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3];

					points.push([
						b[0] * x0 + b[1] * n[0] + b[2] * n[2] + b[3] * n[4],
						b[0] * y0 + b[1] * n[1] + b[2] * n[3] + b[3] * n[5]
					]);
				}
			else if (command !== 'Z') points.push([n[0], n[1]]);
		}

		return points;
	}

	const OUTLINES = PATHS.map(outline);

	/**
	 * The two gaps where the icon, at top y, comes nearest the circle crop: the icon's outermost corners seen from the
	 * centre, nearest first, each with the circle that fills the gap, touching the crop and the corner.
	 */
	function gaps(y: number) {
		const c = placement.size / 2;
		const corners = OUTLINES.flatMap((points) => {
			const at = points.map(
				([px, py]): Point => [placement.x + px * placement.scale, y + py * placement.scale]
			);
			const distance = at.map(([px, py]) => Math.hypot(px - c, py - c));

			return at.filter(
				(_, i) => distance[i] >= distance.at(i - 1)! && distance[i] >= distance[(i + 1) % at.length]
			);
		});
		const nearest: { cx: number; cy: number; gap: number }[] = [];

		for (const [px, py] of corners
			.map((p) => ({ d: Math.hypot(p[0] - c, p[1] - c), p }))
			.sort((a, b) => b.d - a.d)
			.map(({ p }) => p)) {
			// a sampled curve can peak twice: keep one gap per corner of the icon
			if (nearest.some((g) => Math.hypot(g.cx - px, g.cy - py) < side / 3)) continue;

			const d = Math.hypot(px - c, py - c);
			const gap = c - d;
			const k = (d + gap / 2) / d;

			nearest.push({ cx: c + (px - c) * k, cy: c + (py - c) * k, gap });

			if (nearest.length === 2) break;
		}

		return nearest;
	}

	const panels = $derived(
		[
			{ label: 'At the exact centre', note: 'the foot comes nearer the circle', y: centred },
			{
				label: `Lifted 1/${lift} of its height`,
				note: 'as in the files: both gaps even',
				y: placement.y
			}
		].map((panel) => ({ ...panel, gaps: gaps(panel.y) }))
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

					<!-- the circle the platforms crop it to -->
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
					{#each panel.gaps as g (g.cx)}
						<circle class="gap" cx={g.cx} cy={g.cy} r={g.gap / 2} />

						<text class="margin" font-size={placement.size * 0.04} x={g.cx} y={g.cy}
							>{round(g.gap)}</text
						>
					{/each}

					<!-- the margins above and below the icon's box -->
					<text
						class="margin"
						font-size={placement.size * 0.045}
						x={placement.size / 2}
						y={panel.y / 2}>{round(panel.y)}</text
					>

					<text
						class="margin"
						font-size={placement.size * 0.045}
						x={placement.size / 2}
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
		The icon fills {Math.round((side / placement.size) * 100)}% of the side. Its weight sits low, in
		the wide foot, and at the exact centre the foot’s corner comes nearer the circle crop than the
		blade’s. The files lift it 1/{lift} of its height, {round(centred - placement.y)} px at {placement.size}
		× {placement.size}: the circles in the two gaps match, so it sits evenly in the circle, and the
		margins above and below the square stay within a few pixels.
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
