<!-- One line at each named weight. -->
<script lang="ts">
	import { specimen } from '../specimen';

	let { text = 'Sphinx of black quartz, judge my vow 2026' }: { text?: string } = $props();
</script>

<section aria-label="Weights" class="waterfall">
	{#each specimen.waterfall as w (w.weight)}
		<div class="wrow">
			<span class="label">{w.weight} · {w.name}</span>

			<p style:font-weight={w.weight}>{text}</p>
		</div>
	{/each}
</section>

<style>
	.waterfall {
		display: grid;
		gap: 12px;
	}
	.wrow {
		display: grid;
		gap: 2px;
	}
	p {
		margin: 0;
		font-family: var(--display);
		font-size: clamp(26px, 4.4vw, 52px);
		line-height: 1.1;
		overflow-wrap: anywhere;
	}
</style>
