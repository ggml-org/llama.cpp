<!--
	Voice: what we say next to what we do not. Plain words and real numbers on one side;
	hype, fear and pride on the other, struck through.
-->
<script lang="ts">
	const pairs = [
		['Runs on your computer, even offline.', 'The most powerful AI ever built.'],
		['4.6 GB. Fits in 8 GB of memory.', 'Blazing fast. Insanely capable.'],
		['Your conversations stay with you.', 'They are watching everything you type.'],
		['Open source, MIT licensed.', 'The only AI you can trust.']
	];
</script>

<div class="voice">
	<p class="head">We say</p>

	<p class="head">We don’t</p>

	{#each pairs as [say, not] (say)}
		<p class="say">{say}</p>

		<p class="not"><s>{not}</s></p>
	{/each}
</div>

<style>
	.voice {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0 24px;
	}

	p {
		margin: 0;
	}

	.head {
		padding-bottom: 10px;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.say,
	.not {
		padding: 12px 0;
		border-top: 1px solid var(--line-default);
		font-size: 17px;
		line-height: 1.35;
	}

	.say {
		font-weight: 600;
	}

	.not {
		color: var(--figure-muted);
	}

	s {
		text-decoration-color: var(--art-cinder);
		text-decoration-thickness: 1.5px;
	}
</style>
