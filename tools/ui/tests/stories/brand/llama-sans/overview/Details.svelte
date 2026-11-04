<!--
	The few details that give Llama Sans its character, each letter drawn large from the specimen's glyph data with
	its angled cuts marked. The cuts are found in the outline itself: short straight edges on a slant.
-->
<script lang="ts">
	import { specimen } from '../lib/specimen';

	const details = [
		{
			ch: 't',
			note: 'Its stem top and foot end are cut on the slant, so each t leans into the next letter.'
		},
		{ ch: 'r', note: 'The arm ends on the same slant, pointing along the line rather than up.' },
		{
			ch: 'f',
			note: 'The hook is cut like the arm of r, so the straight-stemmed letters read as one family.'
		},
		{
			ch: 'y',
			note: 'A small notch where the stroke meets the tail keeps the join crisp at text sizes.'
		}
	];
	const { grid } = specimen;
	const cells = grid.cells['400'];

	type Point = [number, number];

	/** The short straight edges on a slant in an SVG path of M, L, H, V, Q and Z commands. */
	function cuts(d: string): [Point, Point][] {
		const out: [Point, Point][] = [];

		let cur: Point = [0, 0];
		let start: Point = [0, 0];

		const line = (to: Point) => {
			const [dx, dy] = [to[0] - cur[0], to[1] - cur[1]];
			const length = Math.hypot(dx, dy);
			const angle = ((((Math.atan2(dy, dx) * 180) / Math.PI) % 180) + 180) % 180;
			const square = [0, 90, 180].some((a) => Math.abs(angle - a) < 5);

			if (length > 12 && length < 120 && !square) out.push([cur, to]);

			cur = to;
		};

		for (const [, cmd, args] of d.matchAll(/([MLHVQZ])([^MLHVQZ]*)/g)) {
			const n = args.trim()
				? args
						.trim()
						.split(/[\s,]+/)
						.map(Number)
				: [];

			if (cmd === 'M') cur = start = [n[0], n[1]];
			else if (cmd === 'L') for (let i = 0; i < n.length; i += 2) line([n[i], n[i + 1]]);
			else if (cmd === 'H') line([n[0], cur[1]]);
			else if (cmd === 'V') line([cur[0], n[0]]);
			else if (cmd === 'Q') cur = [n[n.length - 2], n[n.length - 1]];
			else if (cmd === 'Z') line(start);
		}

		return out;
	}

	const shown = details.map((x) => {
		const cell = cells.find((c) => c.ch === x.ch)!;

		return {
			...x,
			cell,
			cuts: cuts(cell.d),
			view: `${cell.dx - 60} 0 ${cell.adv + 120} ${grid.height}`,
			x0: cell.dx - 60,
			x1: cell.dx + cell.adv + 60
		};
	});
</script>

<div class="details">
	{#each shown as x (x.ch)}
		<figure>
			<svg aria-hidden="true" viewBox={x.view}>
				<line class="base" x1={x.x0} x2={x.x1} y1={grid.guides.base} y2={grid.guides.base} />

				<line class="base" x1={x.x0} x2={x.x1} y1={grid.guides.x} y2={grid.guides.x} />

				<path d={x.cell.d} />

				{#each x.cuts as [a, b], i (i)}
					<line class="cut" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />

					<circle class="ring" cx={(a[0] + b[0]) / 2} cy={(a[1] + b[1]) / 2} r="78" />
				{/each}
			</svg>

			<figcaption><strong>{x.ch}</strong>{x.note}</figcaption>
		</figure>
	{/each}
</div>

<style>
	.details {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
		gap: 16px;
	}
	figure {
		display: grid;
		gap: 12px;
		align-content: start;
		margin: 0;
	}
	svg {
		display: block;
		width: 100%;
		aspect-ratio: 1 / 1.1;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--sheet);
	}
	path {
		fill: var(--ink);
	}
	.base {
		stroke: var(--rule);
		stroke-width: 3;
	}
	.cut {
		stroke: var(--mark);
		stroke-width: 12;
		stroke-linecap: round;
	}
	.ring {
		fill: none;
		stroke: var(--mark);
		stroke-width: 4;
		stroke-dasharray: 10 10;
	}
	figcaption {
		font: 400 14px / 1.5 var(--ui);
		color: var(--muted);
	}
	figcaption strong {
		display: block;
		font: 600 16px / 1.25 var(--ui);
		color: var(--ink);
	}
</style>
