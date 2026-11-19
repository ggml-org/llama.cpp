<!--
	How the icon sits in a circle crop (the avatars, the PWA icon), animated in steps: the icon holds at the exact
	centre, lifts, is nudged left to where the files place it, is measured, and eases back, on a loop. Below it, a tile
	for each step, as in the logo overview's origin: each lights up once the animation reaches it, and a click moves
	the animation there (with reduced motion it doesn't loop, and a click shows the step at once). The avatar shows light and dark, then measured as it moves: the icon's box, the
	centre lines, the margins all round, and a circle in the gap between the crop and each of the icon's outer points,
	touching both (its rounded corners, its sharp corners and the tips at its middle). The files' placement comes from
	the file itself (its translate and scale), so the figure follows the files.
-->
<script lang="ts">
	import Figure from '../logo-showcase/lib/components/Figure.svelte';
	import Figures from '../logo-showcase/lib/components/Figures.svelte';
	import { PATHS, SIZE } from '../logo-showcase/lib/logo';
	import { untrack } from 'svelte';
	import { cubicInOut } from 'svelte/easing';
	import { prefersReducedMotion, Tween } from 'svelte/motion';

	interface Props {
		/** the canvas's side and the icon's placement in it, as in its SVG */
		placement: { scale: number; size: number; x: number; y: number };
	}

	let { placement }: Props = $props();

	const side = $derived(SIZE * placement.scale);
	const centred = $derived((placement.size - side) / 2);
	// the files' lift and nudge, as fractions of the icon's size: 1/40 up and 1/100 left in the brand's files
	const lift = $derived(Math.round(side / (centred - placement.y)));
	const nudge = $derived(Math.round(side / (centred - placement.x)));

	type Point = [number, number];

	/**
	 * A path's outer corners, in the icon's own 600 units: its rounded corner (its curve's control points), its sharp
	 * one (its corner farthest from the icon's centre) and its tip at the middle (its corner farthest to the side).
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

		return {
			curve,
			sharp: points.reduce((far, p) => (reach(p) > reach(far) ? p : far)),
			tip: points.reduce((far, p) =>
				Math.abs(p[0] - SIZE / 2) > Math.abs(far[0] - SIZE / 2) ? p : far
			)
		};
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
		const outward = (v: number[]) => {
			const d = Math.hypot(v[0], v[1]);

			return fill(v, [v[0] / d, v[1] / d]);
		};

		return CORNERS.flatMap(({ curve: [p0, p1, p2, p3], sharp, tip }, i) => {
			const apex = [0, 1].map((k) => (p0[k] + 3 * p1[k] + 3 * p2[k] + p3[k]) / 8);
			const tangent = [0, 1].map((k) => p3[k] + p2[k] - p1[k] - p0[k]);
			const length = Math.hypot(tangent[0], tangent[1]);

			let n = [tangent[1] / length, -tangent[0] / length];

			if (n[0] * (apex[0] - SIZE / 2) + n[1] * (apex[1] - SIZE / 2) < 0) n = [-n[0], -n[1]];

			return [
				{ ...fill(onCanvas(apex), n), kind: 'rounded', name: NAMES[i] },
				{ ...outward(onCanvas(sharp)), kind: 'sharp', name: NAMES[i] },
				{ ...outward(onCanvas(tip)), kind: 'middle', name: NAMES[i] }
			];
		});
	}

	/** For each kind of corner, whether the pair is even (within half a pixel) or which comes nearer the crop. */
	function balance(g: ReturnType<typeof gaps>) {
		return ['rounded', 'sharp', 'middle']
			.map((kind) => {
				const [a, b] = g.filter((gap) => gap.kind === kind);

				if (Math.abs(a.gap - b.gap) < 0.5) return `${kind} even`;

				return `${kind}: the ${a.gap < b.gap ? a.name : b.name}’s nearer`;
			})
			.join(' · ');
	}

	// the steps, in order: centred, lifted, nudged left to the files' placement, then measured there
	const STEPS = ['centred', 'lifted', 'nudged', 'measured'] as const;
	// how long each step holds before the next
	const HOLDS = [1500, 900, 900, 4500];

	// how far the icon has been lifted and nudged, from the exact centre (0) to the files' placement (1); the measures
	// draw on only once it has arrived, and clear before it leaves
	const lifted = new Tween(0, { duration: 1000, easing: cubicInOut });
	const nudged = new Tween(0, { duration: 900, easing: cubicInOut });
	let step = $state(0);
	let measured = $state(false);
	// where the loop (re)starts: a click on a tile moves it there
	let from = $state({ at: 0 });
	const uid = $props.id();

	const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

	// the loop that's running: a click starts another, and a stale one stops at its next await
	let run = 0;

	/** Moves the icon to step i (lifted from the second, nudged from the third, measured at the last), then shows it. */
	async function show(i: number, id: number, duration?: number) {
		const options = duration === undefined ? undefined : { duration };

		if (i < 3 && measured) {
			measured = false;

			if (duration !== 0) await wait(500);
		}

		if (id !== run) return;

		await Promise.all([lifted.set(i >= 1 ? 1 : 0, options), nudged.set(i >= 2 ? 1 : 0, options)]);

		if (id !== run) return;

		step = i;
		measured = i === 3;
	}

	$effect(() => {
		const start = from.at;
		const id = untrack(() => ++run);

		if (prefersReducedMotion.current) {
			untrack(() => show(start, id, 0));

			return;
		}

		untrack(async () => {
			for (let i = start; id === run; i = (i + 1) % STEPS.length) {
				await show(i, id);

				if (id !== run) break;

				await wait(HOLDS[i]);
			}
		});

		// unmounted, or restarted: this loop is stale
		return () => run++;
	});

	/** The icon's top left with it lifted and nudged by l and n (0 to 1 of the files' lift and nudge). */
	const at = (l: number, n: number) => ({
		x: centred + (placement.x - centred) * n,
		y: centred + (placement.y - centred) * l
	});

	const shown = $derived.by(() => {
		const { x, y } = at(lifted.current, nudged.current);
		const g = gaps(x, y);
		const label = [
			'At the exact centre',
			`Lifted 1/${lift} of its height`,
			`Lifted 1/${lift} and nudged 1/${nudge} left, as in the files`,
			`Lifted 1/${lift} and nudged 1/${nudge} left, as in the files`
		][step];

		return { gaps: g, key: 'shown', label, note: balance(g), x, y };
	});

	// each tile's drawing, at its step's placement
	const tiles = $derived(STEPS.map((name, i) => ({ name, ...at(i >= 1 ? 1 : 0, i >= 2 ? 1 : 0) })));
	const titles = { centred: 'Centred', lifted: 'Lifted', measured: 'Measured', nudged: 'Nudged' };

	const round = (n: number) => Math.round(n * 10) / 10;
</script>

<figure aria-label="Icon placements" class="m-0 grid min-w-0 gap-6">
	<!-- the placement as it moves: the avatar light and dark, small as it shows, then measured -->
	{#each [shown] as panel (panel.key)}
		<section class="grid gap-4">
			<div class="grid gap-1">
				<h3 class="text-sm font-medium">{panel.label}</h3>

				<p class={['fade-note text-xs text-muted-foreground', measured && 'shown']}>{panel.note}</p>
			</div>

			<div class="flex flex-wrap items-center gap-8">
				{#each [['#FAFAFA', '#111111'], ['#111111', '#FAFAFA']] as [ground, ink] (ground)}
					<svg
						class="block size-24 shrink-0 rounded-full border border-border/60"
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

				<div class="w-full sm:ml-8 sm:w-[calc((100%-3rem)/3)]">
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

						<!-- the measures: drawn on once the icon has arrived, in turn, and cleared before it moves back -->
						<g class={['measures', measured && 'shown']}>
							<!-- the icon's box, dashed, revealed by a solid copy drawn on in its mask, then its centre lines -->
							<mask id="{uid}-box">
								<rect
									class="draw"
									fill="none"
									height={side}
									pathLength="1"
									stroke="white"
									stroke-width={placement.size / 40}
									width={side}
									x={panel.x}
									y={panel.y}
								/>
							</mask>

							<rect
								class="guide"
								height={side}
								mask="url(#{uid}-box)"
								width={side}
								x={panel.x}
								y={panel.y}
							/>

							<line
								style:--delay="350ms"
								class="guide solid draw"
								pathLength="1"
								stroke-width={placement.size / 320}
								x1={panel.x}
								x2={panel.x + side}
								y1={panel.y + side / 2}
								y2={panel.y + side / 2}
							/>

							<line
								style:--delay="450ms"
								class="guide solid draw"
								pathLength="1"
								stroke-width={placement.size / 320}
								x1={panel.x + side / 2}
								x2={panel.x + side / 2}
								y1={panel.y}
								y2={panel.y + side}
							/>

							<!-- the gaps between the crop and the icon's outer corners, each filled by a circle touching both -->
							{#each panel.gaps as g, i (g.kind + g.name)}
								<circle
									style:--delay="{600 + i * 90}ms"
									class={['gap draw', g.kind]}
									cx={g.cx}
									cy={g.cy}
									pathLength="1"
									r={g.gap / 2}
									stroke-width={placement.size / 320}
								/>

								<text
									style:--delay="{1100 + i * 60}ms"
									class="margin fade"
									font-size={placement.size * 0.04}
									x={g.cx}
									y={g.cy}>{round(g.gap)}</text
								>
							{/each}

							<!-- the margins round the icon's box: above and below in the square's corners, outside the crop -->
							{#each [{ at: [0.08, panel.y / 2], value: panel.y }, { at: [0.92, (placement.size + panel.y + side) / 2], value: placement.size - panel.y - side }] as m (m.at[0])}
								<text
									style:--delay="1300ms"
									class="margin fade"
									font-size={placement.size * 0.045}
									x={placement.size * m.at[0]}
									y={m.at[1]}>{round(m.value)}</text
								>
							{/each}

							{#each [{ value: panel.x, x: panel.x / 2 }, { value: placement.size - panel.x - side, x: (placement.size + panel.x + side) / 2 }] as m (m.x)}
								<text
									style:--delay="1300ms"
									class="margin fade"
									font-size={placement.size * 0.045}
									x={m.x}
									y={placement.size * 0.42}>{round(m.value)}</text
								>
							{/each}
						</g>
					</svg>
				</div>
			</div>
		</section>
	{/each}

	<!-- the steps, as tiles: each lights up once the animation reaches it, and a click moves the animation there -->
	<Figures layout="steps">
		{#each tiles as tile, i (tile.name)}
			<Figure
				action="Show {titles[tile.name]} in the animation"
				dim={i > step}
				n={String(i + 1).padStart(2, '0')}
				onclick={() => (from = { at: i })}
				title={titles[tile.name]}
				view="0 0 {placement.size} {placement.size}"
			>
				{#snippet drawing()}
					<circle
						class="tile-crop"
						cx={placement.size / 2}
						cy={placement.size / 2}
						r={placement.size / 2 - 1}
					/>

					<line
						class="tile-axis"
						x1="0"
						x2={placement.size}
						y1={placement.size / 2}
						y2={placement.size / 2}
					/>

					<line
						class="tile-axis"
						x1={placement.size / 2}
						x2={placement.size / 2}
						y1="0"
						y2={placement.size}
					/>

					<g
						fill="var(--l-shape)"
						transform="translate({tile.x} {tile.y}) scale({placement.scale})"
					>
						{#each PATHS as d (d)}<path {d} />{/each}
					</g>

					{#if tile.name === 'measured'}
						{#each gaps(tile.x, tile.y) as g (g.kind + g.name)}
							<circle class={['gap tile-gap', g.kind]} cx={g.cx} cy={g.cy} r={g.gap / 2} />
						{/each}
					{:else}
						<rect class="tile-box" height={side} width={side} x={tile.x} y={tile.y} />
					{/if}
				{/snippet}

				{#if tile.name === 'centred'}
					At the exact centre of the square, the icon reads low: its weight sits in the wide foot.
				{:else if tile.name === 'lifted'}
					Up 1/{lift} of its height, it looks centred.
				{:else if tile.name === 'nudged'}
					Left 1/{nudge} of its width: where the files place it, in the avatars and the PWA icon.
				{:else}
					Each gap between the crop and the icon's outer points, filled by a circle touching both.
				{/if}
			</Figure>
		{/each}
	</Figures>

	<figcaption class="text-xs text-muted-foreground">
		The icon fills {Math.round((side / placement.size) * 100)}% of the side; the numbers are pixels
		at {placement.size}
		× {placement.size}. Each circle fills the gap between the circle crop and one of the icon’s
		outer points: orange at the rounded corners (the blade’s top left, the foot’s bottom right),
		grey at the sharp corners (the blade’s top right, the foot’s bottom left), blue at the tips in
		the middle (the blade’s left, the foot’s right). No placement evens all three pairs, since the
		tips are even only without a sideways nudge; lifted 1/{lift} and nudged 1/{nudge} left, the files
		keep every pair within a few pixels.
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
	}
	.gap.sharp {
		fill: color-mix(in srgb, #a3a3a3 14%, transparent);
		stroke: #a3a3a3;
	}
	.gap.middle {
		fill: color-mix(in srgb, #3b82f6 10%, transparent);
		stroke: #3b82f6;
	}
	/* the tiles' marks: the crop and its centre lines, faint, and the icon's box */
	.tile-crop,
	.tile-axis {
		fill: none;
		stroke: var(--l-line);
		stroke-width: 1px;
		vector-effect: non-scaling-stroke;
	}
	.tile-axis {
		stroke-dasharray: 4 3;
	}
	.tile-box {
		fill: none;
		stroke: #ee6c1e;
		stroke-dasharray: 4 3;
		stroke-width: 1px;
		vector-effect: non-scaling-stroke;
	}
	.tile-gap {
		stroke-width: 1px;
		vector-effect: non-scaling-stroke;
	}
	.margin {
		dominant-baseline: middle;
		fill: #ee6c1e;
		font-family: var(--font-mono, monospace);
		text-anchor: middle;
	}

	/* drawn on: a stroke traced from nothing along its length, then the fills and numbers faded in, each after its
	   --delay; cleared quicker, all at once */
	.measures .draw {
		stroke-dasharray: 1;
		stroke-dashoffset: 1;
		vector-effect: none;
		transition: stroke-dashoffset 300ms ease-in;
	}
	.measures.shown .draw {
		stroke-dashoffset: 0;
		transition: stroke-dashoffset 700ms cubic-bezier(0.65, 0, 0.35, 1) var(--delay, 0ms);
	}
	.measures .gap {
		fill-opacity: 0;
		transition:
			stroke-dashoffset 300ms ease-in,
			fill-opacity 250ms ease-in;
	}
	.measures.shown .gap {
		fill-opacity: 1;
		transition:
			stroke-dashoffset 700ms cubic-bezier(0.65, 0, 0.35, 1) var(--delay, 0ms),
			fill-opacity 500ms ease-out calc(var(--delay, 0ms) + 500ms);
	}
	.measures .fade,
	.fade-note {
		opacity: 0;
		transition: opacity 250ms ease-in;
	}
	.measures.shown .fade,
	.fade-note.shown {
		opacity: 1;
		transition: opacity 400ms ease-out var(--delay, 1300ms);
	}
	@media (prefers-reduced-motion: reduce) {
		.measures *,
		.fade-note {
			transition: none !important;
		}
	}
</style>
