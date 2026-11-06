<!--
	How the icon sits in the avatars: the same square twice, the icon set at the exact centre and as the files place
	it, lifted a little, with its box, both centre lines and the margins above and below drawn in. Its weight is low
	(the wide foot), so at the exact centre it reads as sitting low; lifted, it reads as centred. The placement comes
	from the avatar file itself (its translate and scale), so the figure follows the files.
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
	// the lift as a fraction of the icon's height: 1/60 in the brand's files
	const lift = $derived(Math.round(side / (centred - placement.y)));

	const panels = $derived([
		{ label: 'At the exact centre', note: 'reads as sitting low', y: centred },
		{
			label: `Lifted 1/${lift} of its height`,
			note: 'as in the files: reads as centred',
			y: placement.y
		}
	]);

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
		the wide foot, so the files lift it 1/{lift} of its height above centre: {round(
			centred - placement.y
		)} px at {placement.size} × {placement.size}. The margins above and below differ, but the mark
		looks evenly placed in the square and in the circle the platforms crop it to.
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
	.margin {
		dominant-baseline: middle;
		fill: #ee6c1e;
		font-family: var(--font-mono, monospace);
		text-anchor: middle;
	}
</style>
