<!--
	The details that give Llama Sans its character, up close: a poster of one letter's anatomy (its outline, points and
	handles), then tight crops of each angled cut and stem joint, found in the specimen's own outlines so they follow
	any change to the letters.
-->
<script lang="ts">
	import { specimen } from '../lib/specimen';
	import { cuts, joins, type Point, points } from './outline';

	const { glyphCount, grid } = specimen;
	const cell = (ch: string) => grid.cells['400'].find((c) => c.ch === ch)!;

	type Detail = {
		at: 'bottom' | 'top' | number;
		ch: string;
		kind: 'cut' | 'join';
		note: string;
		title: string;
	};

	const details: Detail[] = [
		{
			at: 'top',
			ch: 't',
			kind: 'cut',
			note: 'The stem top is cut on a slant, so the t leans into what follows.',
			title: 't · stem top'
		},
		{
			at: 'bottom',
			ch: 't',
			kind: 'cut',
			note: 'The foot ends on the same kind of cut, closing the letter off without a spur.',
			title: 't · foot'
		},
		{
			at: 0,
			ch: 'r',
			kind: 'cut',
			note: 'The arm points along the line instead of up, and gives r its pace.',
			title: 'r · arm'
		},
		{
			at: 0,
			ch: 'f',
			kind: 'cut',
			note: 'The hook is cut like the arm of r, so the straight-stemmed letters agree.',
			title: 'f · hook'
		},
		{
			at: 0,
			ch: 'y',
			kind: 'cut',
			note: 'A small notch where the stroke meets the tail keeps the join crisp at text sizes.',
			title: 'y · notch'
		},
		{
			at: 'top',
			ch: 'n',
			kind: 'join',
			note: 'Where the shoulder leaves the stem, the notch cuts a little deeper to keep the join open.',
			title: 'n · shoulder'
		},
		{
			at: 'top',
			ch: 'g',
			kind: 'join',
			note: 'The bowl meets the stem in a clean cut, not a blob of ink.',
			title: 'g · bowl'
		},
		{
			at: 'bottom',
			ch: 'd',
			kind: 'join',
			note: 'Low joins open the same way, so d, b, q and u keep an even colour.',
			title: 'd · foot'
		}
	];

	const SIZE = 300; // close-up crop, in grid units

	const closeups = details.map((x) => {
		const c = cell(x.ch);
		const found = x.kind === 'cut' ? cuts(c.d) : joins(c.d).map((p) => ({ a: p, angle: 0, b: p }));
		const pick =
			typeof x.at === 'number'
				? found[x.at]
				: found.reduce((best, f) =>
						(
							x.at === 'top'
								? f.a[1] + f.b[1] < best.a[1] + best.b[1]
								: f.a[1] + f.b[1] > best.a[1] + best.b[1]
						)
							? f
							: best
					);
		const centre: Point = [(pick.a[0] + pick.b[0]) / 2, (pick.a[1] + pick.b[1]) / 2];

		return {
			...x,
			anatomy: points(c.d),
			angle: x.kind === 'cut' ? `${pick.angle}°` : 'joint',
			cell: c,
			centre,
			pick,
			view: `${centre[0] - SIZE / 2} ${centre[1] - SIZE / 2} ${SIZE} ${SIZE}`
		};
	});

	const poster = cell('g');
	const posterPoints = points(poster.d);
	const posterView = `${poster.dx - 70} ${grid.guides.asc + 90} ${poster.adv + 140} ${grid.guides.dsc - grid.guides.asc + 40}`;
</script>

<div class="closeups">
	<figure class="anatomy">
		<p class="spec">
			Llama Sans<br />UI · Display<br />&nbsp;&nbsp;Draft 0.6<br />&nbsp;&nbsp;Glyphs: {glyphCount}<br
			/>&nbsp;&nbsp;OFL 1.1
		</p>

		<svg aria-hidden="true" viewBox={posterView}>
			{#each posterPoints.handles as [a, b], i (i)}
				<line class="handle" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />
			{/each}

			<path d={poster.d} />

			{#each posterPoints.nodes as p, i (i)}
				<rect class="node" height="16" width="16" x={p[0] - 8} y={p[1] - 8} />
			{/each}
		</svg>

		<figcaption>
			The g, point by point: quadratic curves, eased toward circles, with every stroke end cut.
		</figcaption>
	</figure>

	{#each closeups as x (x.title)}
		<figure class="closeup">
			<div class="crop">
				<svg aria-hidden="true" viewBox={x.view}>
					<path class="fill" d={x.cell.d} />

					{#each x.anatomy.handles as [a, b], i (i)}
						<line class="handle" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />
					{/each}

					<path class="edge" d={x.cell.d} />

					{#if x.kind === 'cut'}
						<line class="cut" x1={x.pick.a[0]} x2={x.pick.b[0]} y1={x.pick.a[1]} y2={x.pick.b[1]} />
					{:else}
						<circle class="ring" cx={x.centre[0]} cy={x.centre[1]} r="44" />
					{/if}

					{#each x.anatomy.nodes as p, i (i)}
						<rect class="node" height="9" width="9" x={p[0] - 4.5} y={p[1] - 4.5} />
					{/each}
				</svg>

				<span class="tag">{x.angle}</span>
			</div>

			<figcaption><strong>{x.title}</strong>{x.note}</figcaption>
		</figure>
	{/each}
</div>

<style>
	.closeups {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 16px;
	}
	figure {
		display: grid;
		gap: 10px;
		align-content: start;
		margin: 0;
	}
	/* the anatomy card, after Geist's poster: always on black, white hairlines and square points */
	.anatomy {
		position: relative;
		grid-column: span 2;
		grid-row: span 2;
		align-content: space-between;
		padding: 24px;
		border-radius: 14px;
		background: var(--ds-neutral-950);
		color: var(--ds-neutral-50);
	}
	.anatomy .spec {
		margin: 0;
		font: 400 15px / 1.25 var(--mono);
		letter-spacing: 0.02em;
		text-transform: uppercase;
	}
	.anatomy svg {
		display: block;
		width: 100%;
		max-height: 520px;
	}
	.anatomy path {
		fill: none;
		stroke: var(--ds-neutral-50);
		stroke-width: 3;
	}
	.anatomy .handle {
		stroke: rgb(255 255 255 / 0.4);
		stroke-width: 2;
	}
	.anatomy .node {
		fill: var(--ds-neutral-950);
		stroke: var(--ds-neutral-50);
		stroke-width: 2.5;
	}
	.anatomy figcaption {
		font: 400 13px / 1.5 var(--ui);
		color: rgb(255 255 255 / 0.55);
	}
	/* the close-ups: the filled letter with its outline and points traced over it */
	.crop {
		position: relative;
		overflow: hidden;
		aspect-ratio: 1;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--sheet);
	}
	.crop svg {
		display: block;
		width: 100%;
		height: 100%;
	}
	.fill {
		fill: var(--ink);
		opacity: 0.1;
	}
	.edge {
		fill: none;
		stroke: var(--ink);
		stroke-width: 2;
	}
	.crop .handle {
		stroke: var(--muted);
		stroke-width: 1.2;
		stroke-dasharray: 4 4;
	}
	.crop .node {
		fill: var(--sheet);
		stroke: var(--ink);
		stroke-width: 1.5;
	}
	.cut {
		stroke: var(--mark);
		stroke-width: 9;
		stroke-linecap: round;
	}
	.ring {
		fill: none;
		stroke: var(--mark);
		stroke-width: 3;
		stroke-dasharray: 7 6;
	}
	.tag {
		position: absolute;
		top: 10px;
		left: 10px;
		padding: 2px 8px;
		border-radius: 999px;
		background: var(--clay-soft);
		color: var(--clay);
		font: 500 11px / 1.5 var(--mono);
	}
	figcaption {
		font: 400 13px / 1.5 var(--ui);
		color: var(--muted);
	}
	figcaption strong {
		display: block;
		font: 600 15px / 1.25 var(--ui);
		color: var(--ink);
	}
	@media (max-width: 900px) {
		.closeups {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
</style>
