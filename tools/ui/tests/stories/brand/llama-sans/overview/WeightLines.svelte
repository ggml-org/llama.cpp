<!-- Every named weight as one large line, after rsms.me/inter: its name and number over a phrase set in it. -->
<script lang="ts">
	import { specimen } from '../lib/specimen';

	let { face = 'var(--headline)' }: { face?: string } = $props();

	const phrases: Record<number, string> = {
		400: 'Your own machine',
		500: 'Quantised to fit',
		600: 'Tokens per second',
		700: 'Local inference'
	};
</script>

<section style:--face={face} aria-label="Weights" class="weights">
	{#each specimen.styles as s (s.weight)}
		<div class="row">
			<p class="meta"><span>{s.name}</span><span>{s.weight}</span></p>

			<p style:font-weight={s.weight} class="line">{phrases[s.weight] ?? s.name}</p>
		</div>
	{/each}
</section>

<style>
	.weights {
		display: grid;
		gap: clamp(20px, 3vw, 40px);
		width: var(--column);
		margin-inline: auto;
	}
	.row {
		display: grid;
		gap: 4px;
	}
	.row > * {
		margin: 0;
	}
	.meta {
		display: flex;
		justify-content: space-between;
		font: 400 14px / 1.25 var(--ui);
		color: var(--muted);
	}
	.line {
		overflow: hidden;
		font: 400 clamp(44px, 8.6vw, 112px) / 1.05 var(--face);
		letter-spacing: -0.01em;
		white-space: nowrap;
	}
</style>
