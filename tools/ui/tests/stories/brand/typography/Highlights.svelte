<!--
	What sets one version of the family apart, as a grid of cards: each one a short sample set in that version (face),
	what it shows and why. Part of the Typography overview.
-->
<script lang="ts">
	export interface Highlight {
		/** the sample, set large in the version's face */
		sample: string;
		/** the setting behind it, e.g. "Tracking +3" */
		spec: string;
		text: string;
		title: string;
	}

	let {
		face,
		items,
		large = false
	}: {
		face: string;
		items: Highlight[];
		/** set the samples larger and bolder, for Display */
		large?: boolean;
	} = $props();
</script>

<div class:large style:--face={face} class="highlights">
	{#each items as h (h.title)}
		<article>
			<p class="sample">{h.sample}</p>

			<div class="body">
				<h3>{h.title}</h3>

				<p>{h.text}</p>

				<p class="spec">{h.spec}</p>
			</div>
		</article>
	{/each}
</div>

<style>
	.highlights {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
		gap: 12px;
	}
	article {
		display: grid;
		grid-template-rows: auto 1fr;
		overflow: hidden;
		border-radius: 14px;
		background: var(--sheet);
	}
	.sample {
		display: grid;
		place-items: center;
		min-height: 120px;
		margin: 0;
		padding: 20px;
		border-bottom: 1px solid var(--rule);
		font: 500 clamp(24px, 2.6vw, 32px) / 1.15 var(--face);
		text-align: center;
		white-space: pre-line;
	}
	.large .sample {
		min-height: 140px;
		font: 700 clamp(30px, 3.4vw, 42px) / 1.15 var(--face);
		letter-spacing: -0.01em;
	}
	.body {
		display: grid;
		gap: 6px;
		align-content: start;
		padding: 16px 18px 18px;
	}
	.body > * {
		margin: 0;
	}
	h3 {
		font: 600 16px / 1.3 var(--ui);
	}
	.body p {
		font: 400 14px / 1.5 var(--ui);
	}
	.body .spec {
		margin-top: 4px;
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
</style>
