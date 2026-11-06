<!-- Llama Sans Display at work: a type card, weights, numerals, posters, a schedule, a cover, stickers and a billboard.
     Type is sized in cqi, so each card keeps its composition at any width. -->
<script lang="ts">
	const weights = [400, 450, 500, 550, 600, 650, 700];
	const ladder = [10, 14, 18, 24, 32, 44];
	const dates: [string, string, string][] = [
		['22 AUG', 'MUNICH (DE)', 'Werksviertel'],
		['04 SEP', 'LOS ANGELES (US)', 'The Lot'],
		['11 SEP', 'NEW YORK (US)', 'Pier 57'],
		['23 OCT', 'AMSTERDAM (NL)', 'Melkweg'],
		['07 NOV', 'WARSAW (PL)', 'Hydrozagadka'],
		['02 DEC', 'TOKYO (JP)', 'Shibuya Stream']
	];
	const stickers: [string, string][] = [
		['1.0', 'ink'],
		['&', 'clay'],
		['+', 'slate'],
		['Aa', 'paper'],
		['Q8', 'clay'],
		['GGUF', 'ink'],
		['70B', 'slate'],
		['2026', 'paper']
	];
</script>

<section aria-label="Llama Sans Display at work" class="gallery display">
	<!-- a specimen card, after the classic type posters -->
	<article class="card wide c-hero type-card">
		<div class="meta">
			<p class="name">Llama Sans<br />Display</p>

			<p>Variable<br />400–700</p>

			<p>by ggml-org</p>
		</div>

		<div class="alpha">
			<p>ABCDEFGHIJKLMN<br />OPQRSTUVWXYZ</p>

			<p>abcdefghijklmn<br />opqrstuvwxyz()&?!</p>

			<p>0123456789.,:;“”/–</p>
		</div>

		<ol aria-label="Sizes" class="ladder">
			{#each ladder as px (px)}<li>
					<span>{px}</span><b style:font-size="{px / 7}cqi">Aa</b>
				</li>{/each}
		</ol>

		<p class="word"><span>Abcdefg</span></p>
	</article>

	<!-- every step of the weight axis -->
	<article class="card c-sand weights">
		<p class="mono-label">One axis · 400 to 700</p>

		<ol>
			{#each weights as w (w)}<li style:font-weight={w}><span>{w}</span>Llama Sans</li>{/each}
		</ol>
	</article>

	<!-- big numerals, solid and outlined -->
	<article class="card c-paper numerals">
		<p class="n solid">128K</p>

		<p class="n outline">128K</p>

		<p class="n slate">128K</p>

		<p class="mono-label">Tokens of context</p>
	</article>

	<!-- a wall of small posters -->
	<article class="card c-ink posters">
		<div class="tile t-sand"><span class="plus">+</span></div>

		<div class="tile t-paper"><span class="q">Q4</span><span class="tag">4-bit</span></div>

		<div class="tile t-paper story">
			<span>Once</span><span>upon</span><span>a</span><span>token</span>
		</div>

		<div class="tile t-blue"><span class="big">70B</span><span class="tag">at home</span></div>
	</article>

	<!-- a tour schedule: one face, two weights -->
	<article class="card c-paper schedule">
		<p class="head">
			<span class="mono-label">Sample tour · 2026</span><span class="mono-label"
				>Display 400 &amp; 700</span
			>
		</p>

		<ol>
			{#each dates as [date, city, venue] (date)}<li>
					<span class="date">{date}</span><b>{city}</b><span class="venue">{venue}</span>
				</li>{/each}
		</ol>
	</article>

	<!-- a magazine cover -->
	<article class="card c-ink cover">
		<p class="masthead">LOCAL</p>

		<p class="issue">No. 04 · November 2026</p>

		<h3 class="lead">Small models, big ideas.</h3>

		<ul class="lines">
			<li>Quantization, explained</li>

			<li>Running 70B at home</li>

			<li>The GGUF story</li>
		</ul>
	</article>

	<!-- a sheet of stickers -->
	<article class="card c-hero stickers">
		{#each stickers as [label, tone], i (i)}<span
				class:long={label.length > 2}
				class="sticker s-{tone}">{label}</span
			>{/each}
	</article>

	<!-- a glyph close-up on its guides: drawn in SVG, where text sits exactly on its baseline -->
	<article class="card c-paper glyphs">
		<svg
			aria-label="The letters a and g on the baseline, x-height and cap height"
			role="img"
			viewBox="0 0 400 300"
		>
			{#each [[230, 'baseline'], [230 - 0.53 * 220, 'x-height 530'], [230 - 0.71 * 220, 'cap 710']] as [y, label] (label)}
				<line class="guide" x1="0" x2="400" y1={y} y2={y} />

				<text class="guide-label" text-anchor="end" x="388" y={+y - 6}>{label}</text>
			{/each}

			<text class="ag" x="24" y="230">ag</text>
		</svg>
	</article>

	<!-- a billboard -->
	<article class="card full c-sand billboard">
		<p class="claim">Run it yourself.</p>

		<p class="by"><span>llama.cpp</span><span>Inference on your own hardware</span></p>
	</article>
</section>

<style>
	.display {
		font-family: var(--display);
	}

	/* type card */
	.type-card {
		display: grid;
		grid-template-columns: 1fr 1.6fr 0.8fr;
		grid-template-rows: auto 1fr;
		gap: 1.2cqi 3cqi;
		padding: 3cqi 3.4cqi;
	}
	.type-card .meta {
		display: grid;
		gap: 1cqi;
		align-content: start;
		font: 600 1.6cqi / 1.15 var(--display);
	}
	.type-card .alpha {
		display: grid;
		gap: 1cqi;
		align-content: start;
		font: 600 1.6cqi / 1.15 var(--display);
	}
	.type-card .alpha p,
	.type-card .meta p {
		padding-bottom: 0.8cqi;
		border-bottom: 1px solid var(--llama-hero-rule);
	}
	.type-card .ladder {
		grid-row: span 2;
		display: grid;
		align-content: space-between;
		padding: 0;
		list-style: none;
	}
	.type-card .ladder li {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		border-bottom: 1px solid var(--llama-hero-rule);
	}
	.type-card .ladder span {
		font: 500 1cqi var(--mono);
		opacity: 0.7;
	}
	.type-card .ladder b {
		font-weight: 500;
		line-height: 1.05;
	}
	.type-card .word {
		grid-column: span 2;
		align-self: end;
		font: 700 10.5cqi / 0.9 var(--display);
	}
	.type-card .word span {
		display: inline-block;
		padding-block: 0.5cqi 1.4cqi;
		border-block: 1px solid var(--llama-hero-rule);
	}

	/* weights */
	.weights {
		display: grid;
		align-content: space-between;
		padding: 6cqi;
	}
	.weights ol {
		display: grid;
		padding: 0;
		list-style: none;
		font-size: 8.2cqi;
		line-height: 1.04;
	}
	.weights li {
		display: flex;
		align-items: baseline;
		gap: 3cqi;
		white-space: nowrap;
	}
	.weights li span {
		width: 7cqi;
		font: 500 2.4cqi var(--mono);
		opacity: 0.75;
	}

	/* numerals */
	.numerals {
		display: grid;
		align-content: center;
		padding: 5cqi 6cqi;
	}
	.numerals .n {
		font: 700 23cqi / 0.86 var(--display);
		font-variant-numeric: tabular-nums;
		letter-spacing: -0.01em;
	}
	.numerals .outline {
		color: transparent;
		-webkit-text-stroke: 0.4cqi var(--cinder);
	}
	.numerals .slate {
		color: var(--slate);
	}
	.numerals .mono-label {
		margin-top: 3cqi;
	}

	/* poster wall */
	.posters {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 2.4cqi;
		padding: 2.4cqi;
	}
	.tile {
		position: relative;
		display: grid;
		overflow: hidden;
		border-radius: 1.6cqi;
		padding: 3cqi;
	}
	.t-sand {
		background: var(--sand);
		color: var(--sand-ink);
		place-items: center;
	}
	.t-paper {
		background: var(--brand-light);
		color: var(--brand-ink);
	}
	.t-blue {
		background: var(--llama-navy);
		color: #fff;
	}
	.tile .plus {
		font: 700 36cqi / 0.6 var(--display);
	}
	.tile .q {
		font: 700 20cqi / 0.8 var(--display);
		align-self: end;
	}
	.tile .tag {
		position: absolute;
		top: 3cqi;
		left: 3cqi;
		font: 500 2.2cqi var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.tile.story {
		justify-items: end;
		align-content: center;
		font: 600 6.6cqi / 0.98 var(--display);
		text-align: right;
	}
	.tile .big {
		font: 700 15cqi / 0.8 var(--display);
		align-self: end;
	}

	/* schedule */
	.schedule {
		display: grid;
		grid-template-rows: auto 1fr;
		gap: 3cqi;
		padding: 5cqi 5.6cqi;
	}
	.schedule .head {
		display: flex;
		justify-content: space-between;
	}
	.schedule .head .mono-label {
		font-size: 2.2cqi;
	}
	.schedule ol {
		display: grid;
		align-content: space-between;
		padding: 0;
		list-style: none;
		font-size: 4.1cqi;
		line-height: 1.15;
	}
	.schedule li {
		display: grid;
		grid-template-columns: 15cqi auto 1fr;
		gap: 2.4cqi;
		align-items: baseline;
		white-space: nowrap;
	}
	.schedule .date {
		font-weight: 400;
		font-variant-numeric: tabular-nums;
	}
	.schedule b {
		font-weight: 700;
	}
	.schedule .venue {
		font-weight: 400;
		opacity: 0.7;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	/* cover */
	.cover {
		display: grid;
		grid-template-rows: auto auto 1fr auto;
		padding: 4cqi 5cqi 5cqi;
		background:
			radial-gradient(120% 80% at 85% 110%, rgb(255 255 255 / 0.08), transparent 60%),
			var(--llama-navy); /* one hue: navy, a little lighter in a corner */
	}
	.cover .masthead {
		font: 700 27cqi / 0.8 var(--display);
		letter-spacing: -0.02em;
	}
	.cover .issue {
		margin-top: 2cqi;
		font: 500 2.4cqi var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		opacity: 0.75;
	}
	.cover .lead {
		align-self: end;
		max-width: 12ch;
		font: 700 9cqi / 0.98 var(--display);
	}
	.cover .lines {
		display: grid;
		gap: 0.6cqi;
		margin-top: 3cqi;
		padding: 0;
		list-style: none;
		font: 500 3.4cqi / 1.15 var(--display);
		opacity: 0.85;
	}

	/* stickers */
	.stickers {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		align-content: center;
		gap: 3cqi;
		padding: 5cqi;
	}
	.sticker {
		display: grid;
		place-items: center;
		aspect-ratio: 1;
		border-radius: 50%;
		font: 700 9cqi / 1 var(--display);
		box-shadow: 0 0.6cqi 1.6cqi rgb(0 0 0 / 0.12);
	}
	.sticker.long {
		font-size: 4.2cqi;
	}
	.s-ink {
		background: var(--brand-ink);
		color: var(--brand-light);
	}
	.s-clay {
		background: var(--cinder);
		color: #fff;
	}
	.s-slate {
		background: var(--slate);
		color: #fff;
	}
	.s-paper {
		background: var(--brand-light);
		color: var(--brand-ink);
	}

	/* glyph close-up */
	.glyphs svg {
		display: block;
		width: 100%;
		height: 100%;
	}
	.glyphs .guide {
		stroke: var(--ds-neutral-400);
		stroke-width: 1;
		opacity: 0.55;
		vector-effect: non-scaling-stroke;
	}
	.glyphs .guide-label {
		font: 500 9px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		fill: var(--ds-neutral-500);
	}
	.glyphs .ag {
		font: 600 220px var(--display);
		fill: var(--brand-ink);
	}

	/* billboard */
	.billboard {
		display: grid;
		align-content: space-between;
		padding: 3.4cqi 4cqi;
	}
	.billboard .claim {
		font: 700 12cqi / 0.92 var(--display);
		letter-spacing: -0.01em;
	}
	.billboard .by {
		display: flex;
		justify-content: space-between;
		gap: 2cqi;
		font: 600 1.9cqi / 1.15 var(--display);
	}
	@media (max-width: 700px) {
		.billboard .claim {
			font-size: 19cqi;
		}
		.billboard .by {
			font-size: 3.6cqi;
		}
		.type-card {
			grid-template-columns: 1fr 1.6fr;
			gap: 3cqi;
			padding: 5cqi;
		}
		.type-card .meta,
		.type-card .alpha {
			font-size: 3.6cqi;
		}
		.type-card .ladder {
			display: none;
		}
		.type-card .word {
			font-size: 24cqi;
		}
	}
</style>
