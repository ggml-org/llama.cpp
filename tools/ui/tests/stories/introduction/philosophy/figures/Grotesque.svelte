<!--
	The grotesque line, from Akzidenz-Grotesk to Llama Sans (every name set in Llama Sans, not in its own face), then
	the letters that make Llama Sans its own, and Llama Mono's cells
-->
<script lang="ts">
	const line = [
		{ name: 'Akzidenz-Grotesk', note: 'Berthold, Berlin', year: '1898' },
		{ name: 'Neue Haas Grotesk', note: 'Haas, Münchenstein; Helvetica from 1960', year: '1957' },
		{ name: 'The New York subway signs', note: 'Unimark’s standards manual', year: '1970' },
		{ name: 'San Francisco', note: 'Apple’s system font', year: '2015' },
		{ name: 'Geist', note: 'Vercel, open source', year: '2023' },
		{ name: 'Llama Sans', note: 'built from Geist', ours: true, year: '2026' }
	];

	const letters = [
		{ glyph: 'a', note: 'tailless a' },
		{ glyph: 't', note: 'narrow t' },
		{ glyph: 'S', note: 'angle-cut ends' },
		{ glyph: 'g', note: 'round bowls' }
	];

	const cells = ['l1I|', '0O', '=>', '!=='];
</script>

<div class="grotesque">
	<ol class="line">
		{#each line as l (l.name)}
			<li class={{ ours: l.ours }}>
				<span class="year">{l.year}</span>

				<span class="name">{l.name}</span>

				<span class="note">{l.note}</span>
			</li>
		{/each}
	</ol>

	<div class="letters">
		{#each letters as l (l.glyph)}
			<div class="letter">
				<span class="glyph">{l.glyph}</span>

				<span class="note">{l.note}</span>
			</div>
		{/each}
	</div>

	<div class="mono">
		{#each cells as c (c)}
			<span class="cells">
				{#each [...c] as ch, i (i)}<span class="cell">{ch}</span>{/each}
			</span>
		{/each}

		<span class="note"
			>Llama Mono: one cell a character, so l, 1, I and | stay apart and columns line up</span
		>
	</div>
</div>

<style>
	.grotesque {
		display: grid;
		gap: clamp(32px, 5vw, 56px);
	}

	.line {
		display: grid;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.line li {
		display: grid;
		grid-template-columns: 4rem minmax(0, 1fr) minmax(0, 1fr);
		gap: 16px;
		align-items: baseline;
		padding-block: 10px;
		border-bottom: 1px solid var(--line-default);
	}

	.year {
		color: var(--figure-muted);
		font: 500 13px var(--mono);
	}

	.name {
		font-size: 20px;
		font-weight: 500;
	}

	.ours .name {
		font-weight: 700;
	}

	.ours .year {
		color: var(--fg-default);
	}

	.note {
		color: var(--figure-muted);
		font-size: 14px;
	}

	.letters {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr));
		gap: 16px;
	}

	.letter {
		display: grid;
		gap: 4px;
	}

	.glyph {
		font-size: clamp(96px, 14vw, 168px);
		font-weight: 600;
		line-height: 1;
	}

	.mono {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 20px;
		align-items: center;
	}

	.cells {
		display: inline-flex;
	}

	.cell {
		display: inline-grid;
		place-items: center;
		width: 0.6em;
		height: 1.25em;
		margin-left: -1px;
		border: 1px solid var(--line-default);
		font: 500 36px var(--mono);
	}

	.mono .note {
		flex-basis: 100%;
	}

	@media (width < 640px) {
		.line li {
			grid-template-columns: 3.5rem 1fr;
		}

		.line .note {
			grid-column: 2;
		}
	}
</style>
