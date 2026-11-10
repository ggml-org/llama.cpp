<!--
	One model that routes to several: the model people pick as a card (its id in Mono, what it is under it), a line
	from it to each model it can send a query to, and those as cards with where each runs (a Mono label), its name and
	when it is picked. The first route is the one the piece is about: its line is the cinder detail, the others are in
	the ground's ink. A motif for a Statement, beside the text on a wide format or above it on a square one.
-->
<script lang="ts">
	interface Route {
		/** where it runs, a short label: Local, Remote */
		label: string;
		name: string;
		/** where it runs and when it is picked */
		note: string;
	}

	interface Props {
		/** the model people pick */
		model: string;
		/** what it is */
		note: string;
		routes: Route[];
	}

	let { model, note, routes }: Props = $props();

	const rows = $derived(routes.length);
</script>

<div class="diagram">
	<div class="card hub">
		<p class="model">{model}</p>

		<p class="note">{note}</p>
	</div>

	<div class="links">
		<svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 100 {rows * 100}">
			{#each routes as route, i (route.name)}
				<path
					class:accent={i === 0}
					d="M0 {rows * 50} C50 {rows * 50} 50 {i * 100 + 50} 100 {i * 100 + 50}"
					vector-effect="non-scaling-stroke"
				/>
			{/each}
		</svg>

		<i style:top="50%" class="dot from"></i>

		{#each routes as route, i (route.name)}
			<i style:top="{((i + 0.5) / rows) * 100}%" class="dot to"></i>
		{/each}
	</div>

	<ol style:--rows={rows} class="routes">
		{#each routes as route (route.name)}
			<li class="card">
				<p class="label">{route.label}</p>

				<p class="name">{route.name}</p>

				<p class="note">{route.note}</p>
			</li>
		{/each}
	</ol>
</div>

<style>
	.diagram {
		display: grid;
		grid-template-columns: auto minmax(calc(var(--u) * 1.6), 1fr) minmax(0, calc(var(--u) * 8.4));
		align-items: stretch;
		width: 100%;
		height: 100%;
	}
	.card {
		display: grid;
		align-content: center;
		gap: calc(var(--u) * 0.3);
		padding: calc(var(--u) * 0.55) calc(var(--u) * 0.7);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--u) * 0.45);
		background: var(--p-card);
	}
	.card p {
		margin: 0;
	}
	/* the model people pick on the routes' middle, as wide as its id */
	.hub {
		align-self: center;
		margin-left: calc(var(--u) * 0.8);
	}
	.model {
		font: 700 calc(var(--u) * 0.96) / 1 var(--mono);
		letter-spacing: -0.03em;
		white-space: nowrap;
	}
	/* the lines start and end at the cards' edges */
	.links {
		position: relative;
	}
	.links svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		fill: none;
		stroke: var(--p-ink);
		stroke-width: calc(var(--u) * 0.07);
	}
	.links path.accent {
		stroke: var(--art-cinder);
	}
	.dot {
		position: absolute;
		width: calc(var(--u) * 0.22);
		height: calc(var(--u) * 0.22);
		border-radius: 50%;
		background: var(--p-ink);
		translate: -50% -50%;
	}
	.from {
		left: 0;
	}
	.to {
		left: 100%;
	}
	/* one row per route, the card centred in it, so each line ends at a card's centre */
	.routes {
		display: grid;
		grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.routes li {
		align-self: center;
		margin-block: calc(var(--u) * 0.2);
	}
	.label {
		color: var(--p-muted);
		font: 500 calc(var(--u) * 0.54) / 1 var(--mono);
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}
	.name {
		margin-top: calc(var(--u) * 0.3) !important;
		font-size: calc(var(--u) * 1.1);
		font-weight: 700;
		letter-spacing: -0.02em;
		line-height: 1;
	}
	.note {
		color: var(--p-muted);
		font-size: calc(var(--u) * 0.62);
		line-height: 1.35;
		text-wrap: balance;
	}
</style>
