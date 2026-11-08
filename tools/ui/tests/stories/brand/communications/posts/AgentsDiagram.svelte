<!--
	Llama serving agents: the app icon with its name and endpoint under it, a cinder line from it to each agent, and
	the agents as cards with how each connects in Mono. The icon keeps clear space all round (the lines start off its
	edge, the name sits below it), its centre on the cards' middle; each line ends at its card's centre. A motif for a
	Statement, beside the text on a wide format.
-->
<script lang="ts">
	import AppIcon from './AppIcon.svelte';

	interface Agent {
		name: string;
		/** how it connects: a setting, a command, a file */
		note: string;
	}

	interface Props {
		agents: Agent[];
		/** the server's address under the icon */
		endpoint: string;
	}

	let { agents, endpoint }: Props = $props();

	const rows = $derived(agents.length);
</script>

<div class="diagram">
	<div class="hub">
		<AppIcon size={2.4} />

		<p class="caption">
			<span class="name">Llama</span>

			<span class="endpoint">{endpoint}</span>
		</p>
	</div>

	<div class="links">
		<svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 100 {rows * 100}">
			{#each agents as agent, i (agent.name)}
				<path
					d="M0 {rows * 50} C50 {rows * 50} 50 {i * 100 + 50} 100 {i * 100 + 50}"
					vector-effect="non-scaling-stroke"
				/>
			{/each}
		</svg>

		<i style:top="50%" class="dot from"></i>

		{#each agents as agent, i (agent.name)}
			<i style:top="{((i + 0.5) / rows) * 100}%" class="dot to"></i>
		{/each}
	</div>

	<ol style:--rows={rows} class="agents">
		{#each agents as agent (agent.name)}
			<li>
				<p class="agent">{agent.name}</p>

				<p class="note">{agent.note}</p>
			</li>
		{/each}
	</ol>
</div>

<style>
	.diagram {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) minmax(0, calc(var(--u) * 10.4));
		align-items: stretch;
		width: 100%;
		height: 100%;
	}
	/* the icon on the cards' middle, a little in from the text beside it; its name and endpoint hang under it, so
	   they don't move it off centre */
	.hub {
		position: relative;
		align-self: center;
		margin-left: calc(var(--u) * 0.8);
	}
	.caption {
		position: absolute;
		top: 100%;
		left: 50%;
		display: grid;
		gap: calc(var(--u) * 0.15);
		justify-items: center;
		margin: calc(var(--u) * 0.55) 0 0;
		translate: -50% 0;
		white-space: nowrap;
	}
	.name {
		font-size: calc(var(--u) * 0.84);
		font-weight: 700;
		letter-spacing: -0.02em;
		line-height: 1;
	}
	.endpoint {
		color: var(--p-muted);
		font: 400 calc(var(--u) * 0.58) / 1 var(--mono);
	}
	/* the lines start half a unit off the icon's edge and end at the cards' edge */
	.links {
		position: relative;
		margin-left: calc(var(--u) * 0.5);
	}
	.links svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		fill: none;
		stroke: var(--art-cinder);
		stroke-width: calc(var(--u) * 0.07);
	}
	.dot {
		position: absolute;
		width: calc(var(--u) * 0.22);
		height: calc(var(--u) * 0.22);
		border-radius: 50%;
		translate: -50% -50%;
	}
	.from {
		left: 0;
		background: var(--art-cinder);
	}
	.to {
		left: 100%;
		background: var(--p-ink);
	}
	/* one row per agent, the card centred in it, so each line ends at a card's centre */
	.agents {
		display: grid;
		grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.agents li {
		display: grid;
		align-content: center;
		gap: calc(var(--u) * 0.3);
		align-self: center;
		margin-block: calc(var(--u) * 0.2);
		padding: calc(var(--u) * 0.55) calc(var(--u) * 0.7);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--u) * 0.45);
		background: var(--p-card);
	}
	.agents p {
		margin: 0;
	}
	.agent {
		font-size: calc(var(--u) * 1);
		font-weight: 700;
		letter-spacing: -0.02em;
		line-height: 1;
	}
	.note {
		color: var(--p-muted);
		font: 400 calc(var(--u) * 0.58) / 1.3 var(--mono);
		text-wrap: balance;
	}
</style>
