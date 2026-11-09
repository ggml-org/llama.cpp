<script lang="ts">
	import { type Specimen, specimen } from '../specimen';

	// weight: a named weight with exported cells; comparing: show the earlier build's outlines; grid: another font's
	// exported grid (Llama Mono's), in place of Llama Sans's
	let {
		comparing,
		grid = specimen.grid,
		weight
	}: { comparing: boolean; grid?: Specimen['grid']; weight: number } = $props();

	const g = grid.guides;
	const lines = [g.base, g.x, g.cap, g.asc, g.dsc];
	const cells = $derived(grid.cells[String(weight)]);
	// cells arrive in group order; split them back into the groups
	const sets = $derived.by(() => {
		let i = 0;

		return grid.groups.map((group) => {
			const part = cells.slice(i, i + group.chars.length);

			i += group.chars.length;

			return { cells: part, title: group.title };
		});
	});
</script>

<div class="glyphs">
	{#each sets as set (set.title)}
		<section class="set">
			<h2 class="label">{set.title}</h2>

			<div class="grid">
				{#each set.cells as c (c.ch)}
					<figure class:changed={!!c.prev} class="cell">
						<svg aria-label={c.ch} role="img" viewBox="0 0 {grid.width} {grid.height}">
							{#each lines as y, i (i)}
								<line class:base={i === 0} x1="0" x2={grid.width} y1={y} y2={y} />
							{/each}

							<line class="sb" x1={c.dx} x2={c.dx} y1="0" y2={grid.height} />

							<line class="sb" x1={c.dx + c.adv} x2={c.dx + c.adv} y1="0" y2={grid.height} />

							<path d={comparing && c.prev ? c.prev : c.d} />
						</svg>

						<figcaption><span>{c.ch}</span><span>{c.adv}</span></figcaption>
					</figure>
				{/each}
			</div>
		</section>
	{/each}
</div>

<style>
	.glyphs {
		display: grid;
		gap: 28px;
	}
	.set {
		display: grid;
		gap: 14px;
	}
	h2 {
		margin: 0;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(116px, 1fr));
		gap: 1px;
	}
	.cell {
		margin: 0;
		min-width: 0;
		display: grid;
		align-content: start;
		background: var(--sheet);
		outline: 1px solid var(--rule);
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		padding: 10px 6px 0;
		box-sizing: border-box;
		overflow: visible;
	}
	path {
		fill: var(--ink);
	}
	line {
		stroke: var(--guide-soft);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	line.base {
		stroke: var(--guide);
	}
	line.sb {
		stroke-dasharray: 2 3;
	}
	/* on phones each cell shows its character only, not its advance width */
	@media (max-width: 600px) {
		figcaption span:last-child {
			display: none;
		}
	}
	figcaption {
		display: flex;
		justify-content: space-between;
		padding: 4px 8px 6px;
		font: 400 10px var(--mono);
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}
	.changed figcaption span:first-child::after {
		content: ' ●';
		color: var(--guide);
		font-size: 8px;
		vertical-align: middle;
	}
</style>
