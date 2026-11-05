<!--
	Specimen boards and posters for Llama Sans, after the classic Swiss specimen boards, in the design system's own
	palette: its blues (a Swiss blue, sky, navy), green, amber and red, the greys, and beige as the warm neutral.
	Light mixes stay within one hue. Fixed to light, so they read the same in both themes. Set in Llama Sans Display
	and UI.
-->
<script lang="ts">
	import '../../../brand/fonts/fonts.css';

	const upper = 'ABCDEFGHIJKLMN\nOPQRSTUVWXYZ';
	const lower = 'abcdefghijklmn\nopqrstuvwxyz()&?!';
	const figures = '1234567890.,;:’”/–';
	const rows = [400, 500, 600, 700, 400, 500, 600, 700];
	const weights = [
		[400, 'Regular'],
		[500, 'Medium'],
		[600, 'SemiBold'],
		[700, 'Bold']
	] as const;
	const wall = [
		{ caption: 'Run it yourself', image: 'sky' },
		{ caption: 'Quantised to fit', image: 'sea' },
		{ caption: 'Tokens per second', image: 'peach' },
		{ caption: 'Small models', image: 'mist' }
	];
</script>

<div class="boards">
	<!-- a specimen board on navy, after the Helvetica board -->
	<article aria-label="Specimen board" class="board wide specimen">
		<div class="facts">
			<p><b>Llama Sans<br />UI</b></p>

			<p><b>Published<br />in 2026</b></p>

			<p><b>For llama.cpp</b></p>
		</div>

		<div class="chars">
			<p>{upper}</p>

			<p>{lower}</p>

			<p>{figures}</p>
		</div>

		<ul class="rows">
			{#each rows as w, i (i)}
				<li>
					<span>The Llama Sans design keeps a quiet, even colour from labels to long reading.</span
					><b style:font-weight={w}>Aa</b>
				</li>
			{/each}
		</ul>

		<p class="hero"><span>Abcdefg</span></p>
	</article>

	<!-- a title board on a cool grey gradient -->
	<article aria-label="Title board" class="board wide title">
		<div class="top">
			<p>{upper}</p>

			<p>{lower}</p>

			<p>{figures}</p>
		</div>

		<p class="name">Llama Sans</p>

		<div class="bottom">
			<p>Designed for the<br />llama.cpp web UI</p>

			<p>Published by ggml<br />in 2026</p>
		</div>
	</article>

	<!-- an alphabet board: facts on the left, huge pale capitals on the right -->
	<article aria-label="Alphabet board" class="board wide alphabet">
		<div class="info">
			<p><b>Llama Sans<br />Display</b><span>ABCDEFGHIJKLMN<br />OPQRSTUVWXYZ</span></p>

			<p><b>Published<br />in 2026</b><span>abcdefghijklmn<br />opqrstuvwxyz()&amp;?!</span></p>

			<p class="aa">
				<span>The Display version tightens tracking and kerning for headlines.</span><b>Aa</b>
			</p>
		</div>

		<p aria-hidden="true" class="big">ABCDE<br />FGHIJK<br />LMNOP<br />QRSTU</p>
	</article>

	<!-- a wall of small posters, each with a colour field "photograph" and registration dots -->
	<article aria-label="Poster wall" class="board wide wall">
		{#each wall as w (w.caption)}
			<div class="mini">
				<i class="dot tl"></i><i class="dot tr"></i><i class="dot bl"></i><i class="dot br"></i>

				<div class="photo {w.image}"></div>

				<p><span>{w.caption}</span><span>26</span></p>
			</div>
		{/each}
	</article>

	<!-- a collage of circles on cream -->
	<article aria-label="Circles" class="board wide circles">
		{#each Array.from({ length: 15 }, (_, i) => i) as i (i)}
			<span class="c c{i}">
				{#if i === 1}<b class="year">20<br />26</b>
				{:else if i === 3}<b class="list">Regular<br />Medium<br />SemiBold<br />Bold</b>
				{:else if i === 4}<b>Aa</b>
				{:else if i === 6}<b>&amp;</b>
				{:else if i === 8}<b class="icon"></b>
				{:else if i === 10}<b>*</b>
				{:else if i === 12}<b class="word">Llama<br />Sans</b>
				{:else if i === 13}<b>+</b>{/if}
			</span>
		{/each}
	</article>

	<!-- a weight stack on navy -->
	<article aria-label="Weights" class="board poster stack">
		<p class="lead">Llama Sans–</p>

		{#each weights as [w, name], i (w)}<p class:sky={i % 2 === 0} style:font-weight={w}>
				{name}
			</p>{/each}

		<p class="foot">Four weights · one axis · 400 to 700</p>
	</article>

	<!-- numerals, solid and outline, on sand -->
	<article aria-label="26" class="board poster numerals">
		{#each [0, 1, 2, 3] as i (i)}<p class:navy={i < 2} class:outline-n={i % 2 === 1}>26</p>{/each}
	</article>

	<!-- an exhibition poster with light beams -->
	<article aria-label="Explorations in light" class="board poster light">
		<p class="t">Explorations in light</p>

		<p class="s">An exhibition of type</p>

		<div class="beams"></div>

		<p class="credit">
			<span>Set in Llama Sans<br />Display &amp; UI</span><span>11<br />26</span>
		</p>
	</article>

	<!-- a quiet classic: a sky image under a short title -->
	<article aria-label="A quiet classic" class="board poster classic">
		<div class="photo sky tall"></div>

		<p class="t">Llama<br />Sans</p>

		<p class="s">A quiet classic</p>
	</article>

	<!-- a text poster on sage -->
	<article aria-label="About Llama Sans" class="board poster sage">
		<p class="label">Llama Sans</p>

		<p class="body">
			The Llama Sans design is a neo-grotesk with an even stroke, round bowls and open, angle-cut
			terminals. It was drawn to carry an interface without asking for attention, and to read as
			calmly in a long article as in a button.
		</p>

		<p class="label end">Draft 0.6 · 975 glyphs</p>
	</article>
</div>

<style>
	.boards {
		/* every colour from the design system's palette: its blues, greens, ambers and reds (the status fills and
		   their steps), the greys, and beige as the warm neutral; mixes stay within one hue */
		--navy:
			radial-gradient(
				ellipse at 15% 10%,
				color-mix(in oklab, var(--art-navy), white 6%) 0%,
				transparent 55%
			),
			linear-gradient(
				160deg,
				color-mix(in oklab, var(--art-navy), black 22%) 0%,
				color-mix(in oklab, var(--art-navy), black 10%) 100%
			);
		--navy-flat: color-mix(in oklab, var(--art-navy), black 10%);
		/* fields: neutral greys and beige, In use's light hero blue (mist) and navy, and one vibrant blue; the
		   artwork colours add the pops (shapes, tiles, numerals, highlighted words) */
		--swiss: linear-gradient(
			160deg,
			color-mix(in oklab, var(--art-blue), white 8%) 0%,
			var(--art-blue) 55%,
			color-mix(in oklab, var(--art-blue), black 10%) 100%
		);
		--swiss-flat: var(--art-blue);
		--sky: var(--art-sky);
		--mist: linear-gradient(
			160deg,
			color-mix(in oklab, var(--art-sky), white 55%) 0%,
			color-mix(in oklab, var(--art-sky), white 68%) 100%
		);
		--grey: linear-gradient(
			160deg,
			var(--neutral-400) 0%,
			var(--neutral-300) 55%,
			var(--neutral-200) 100%
		);
		--cream: var(--beige-100);
		--sand: var(--beige-50);
		--sage: var(--art-amber); /* the text poster: a warm, vibrant pop */
		--teal: var(--art-green);
		--saffron: var(--art-amber);
		--pink: var(--art-orange-light); /* a light orange in place of pink: no red */
		--vermilion: var(
			--art-orange
		); /* the accent orange; the brand orange stays with the logotype */
		--ink: var(--beige-900);
		--display: 'Llama Sans', var(--font-sans);
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 14px;
	}
	@media (max-width: 900px) {
		.boards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.boards :is(p, ul) {
		margin: 0;
	}
	.board {
		position: relative;
		display: grid;
		container-type: inline-size;
		overflow: hidden;
		box-sizing: border-box;
		border-radius: 8px;
		font-family: var(--display);
	}
	.wide {
		grid-column: span 2;
		aspect-ratio: 16 / 10;
		padding: 5%;
	}
	.poster {
		aspect-ratio: 3 / 4;
		padding: 8%;
	}

	/* specimen board */
	.specimen {
		grid-template-columns: 0.8fr 1.3fr 1.6fr;
		grid-template-rows: 1fr auto;
		gap: 0 5%;
		background: var(--navy); /* In use's navy */
		color: #fff;
	}
	.specimen .facts {
		display: grid;
		align-content: start;
		gap: 4cqi;
		font: 700 2.6cqi / 1.05 var(--display);
	}
	.specimen .chars {
		display: grid;
		align-content: start;
		gap: 3.6cqi;
		font: 600 2.4cqi / 1.1 var(--display);
		white-space: pre-line;
	}
	.specimen .rows {
		display: grid;
		gap: 0.9cqi;
		padding: 0;
		list-style: none;
	}
	.specimen .rows li {
		display: flex;
		gap: 2cqi;
		justify-content: space-between;
		align-items: center;
		padding-top: 0.6cqi;
		border-top: 1px solid rgb(255 255 255 / 0.3);
	}
	.specimen .rows span {
		max-width: 60%;
		font: 400 1.05cqi / 1.3 var(--font-sans);
		opacity: 0.75;
	}
	.specimen .rows b {
		font: 400 2.6cqi / 1 var(--display);
		color: #fff;
	}
	.specimen .hero {
		grid-column: 1 / 3;
		align-self: end;
		padding: 0 1.4cqi;
		border-block: 1px solid rgb(255 255 255 / 0.45);
		font: 700 10cqi / 1.05 var(--display);
		letter-spacing: -0.02em;
	}

	/* title board */
	.title {
		grid-template-rows: auto 1fr auto;
		background:
			repeating-linear-gradient(90deg, rgb(255 255 255 / 0.45) 0 1px, transparent 1px 8.33%),
			var(--mist); /* llama.app's light hero blue, as in In use */
		color: var(--art-navy);
	}
	.title .top,
	.title .bottom {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 4cqi;
		font: 600 1.9cqi / 1.15 var(--display);
		color: var(--art-navy);
		white-space: pre-line;
	}
	.title .bottom {
		grid-template-columns: 1fr 1fr 1fr;
	}
	.title .bottom p:first-child {
		grid-column: 2;
	}
	.title .name {
		align-self: center;
		font: 400 13cqi / 1 var(--display);
		letter-spacing: -0.03em;
	}

	/* alphabet board */
	.alphabet {
		grid-template-columns: 1fr 1fr;
		gap: 4%;
		background: var(--beige-200);
		color: var(--ink);
	}
	.alphabet .info {
		display: grid;
		align-content: space-between;
	}
	.alphabet .info p {
		display: grid;
		grid-template-columns: 1fr 1.4fr;
		gap: 3cqi;
		padding-bottom: 2.4cqi;
		border-bottom: 1px solid rgb(15 31 51 / 0.2);
		font: 700 2.2cqi / 1.15 var(--display);
	}
	.alphabet .info span {
		font-weight: 600;
	}
	.alphabet .aa {
		align-items: end;
		border: 0 !important;
	}
	.alphabet .aa span {
		font: 400 1.3cqi / 1.4 var(--font-sans);
	}
	.alphabet .aa b {
		justify-self: end;
		font: 400 5cqi / 1 var(--display);
	}
	.alphabet .big {
		align-self: center;
		margin-right: -12% !important;
		font: 700 9.5cqi / 0.98 var(--display);
		color: #fff;
		letter-spacing: -0.01em;
	}

	/* poster wall */
	.wall {
		grid-template-columns: repeat(4, 1fr);
		gap: 3%;
		align-items: center;
		background: var(--cream);
	}
	.mini {
		position: relative;
		display: grid;
		gap: 3cqi;
		padding: 2.4cqi;
		background: var(--beige-50);
		box-shadow: 0 0.4cqi 1.2cqi rgb(0 0 0 / 0.06);
	}
	.mini .photo {
		aspect-ratio: 3 / 4;
	}
	.mini p {
		display: flex;
		justify-content: space-between;
		font: 600 1.2cqi / 1.2 var(--font-sans);
		color: var(--ink);
	}
	.dot {
		position: absolute;
		width: 0.8cqi;
		height: 0.8cqi;
		border-radius: 50%;
		background: var(--vermilion);
	}
	.tl {
		top: 0.8cqi;
		left: 0.8cqi;
	}
	.tr {
		top: 0.8cqi;
		right: 0.8cqi;
	}
	.bl {
		bottom: 0.8cqi;
		left: 0.8cqi;
	}
	.br {
		bottom: 0.8cqi;
		right: 0.8cqi;
	}
	.photo.sky {
		background: linear-gradient(
			180deg,
			color-mix(in oklab, var(--art-sky), black 6%) 0%,
			var(--art-sky) 100%
		);
	}
	.photo.sea {
		background: linear-gradient(
			180deg,
			var(--art-green) 0%,
			color-mix(in oklab, var(--art-green), black 10%) 100%
		);
	}
	.photo.peach {
		background: linear-gradient(
			170deg,
			color-mix(in oklab, var(--art-orange), white 12%) 0%,
			var(--art-orange) 100%
		);
	}
	.photo.mist {
		background: linear-gradient(
			180deg,
			color-mix(in oklab, var(--art-amber), white 12%) 0%,
			var(--art-amber) 100%
		); /* saffron */
	}

	/* circles */
	.circles {
		grid-template-columns: repeat(5, 1fr);
		grid-template-rows: repeat(3, 1fr);
		gap: 2.5%;
		place-items: center;
		background: var(--cream);
	}
	.c {
		display: grid;
		place-items: center;
		width: 80%;
		aspect-ratio: 1;
		border-radius: 50%;
		background: var(--beige-50);
		color: var(--ink);
		container-type: inline-size;
	}
	.c b {
		font: 700 42cqi / 1 var(--display);
	}
	.c .year {
		font-size: 30cqi;
		line-height: 0.85;
	}
	.c .list {
		font: 600 12cqi / 1.1 var(--display);
	}
	.c .word {
		font: 700 20cqi / 0.9 var(--display);
	}
	.c1 {
		background: var(--saffron);
	}
	.c3 {
		background: var(--sky);
	}
	.c4 {
		background: var(--teal);
		color: #fff;
	}
	.c6 {
		background: var(--swiss-flat);
		color: #fff;
	}
	.c7 {
		background: repeating-linear-gradient(0deg, var(--navy-flat) 0 6%, var(--sky) 6% 12%);
	}
	.c8 {
		background: var(--ink);
		color: #fff;
	}
	.c10 {
		background: var(--pink);
	}
	.c11 {
		background: radial-gradient(circle at 35% 35%, var(--neutral-400), var(--neutral-500));
	}
	.c12 {
		background: var(--sky);
	}
	.c13 {
		background: var(--vermilion);
		color: #fff;
	}

	/* weight stack */
	.stack {
		align-content: start;
		background: var(--swiss); /* the one vibrant field: the artwork blue */
		color: #fff;
	}
	.stack p {
		font: 400 13cqi / 1 var(--display);
		letter-spacing: -0.02em;
	}
	.stack .lead {
		font-weight: 700;
	}
	.stack .sky {
		color: var(--art-orange-light); /* a light orange, readable on the dark field */
	}
	.stack .foot {
		align-self: end;
		margin-top: 8cqi;
		font: 500 3.4cqi / 1.3 var(--font-sans);
		letter-spacing: 0;
		opacity: 0.7;
	}

	/* numerals */
	.numerals {
		align-content: center;
		justify-items: center;
		background: var(--cream);
	}
	.numerals p {
		font: 700 34cqi / 0.86 var(--display);
		color: var(--vermilion);
		letter-spacing: -0.03em;
	}
	.numerals .navy {
		color: var(--swiss-flat);
	}
	.numerals .outline-n {
		color: transparent;
		-webkit-text-stroke: 0.6cqi var(--swiss-flat);
	}
	.numerals .outline-n:not(.navy) {
		-webkit-text-stroke-color: var(--vermilion);
	}

	/* exhibition poster */
	.light {
		grid-template-rows: auto auto 1fr auto;
		background: var(--neutral-950);
		color: var(--neutral-100);
	}
	.light .t {
		font: 600 7.4cqi / 1.05 var(--display);
	}
	.light .s {
		justify-self: end;
		margin-top: 3cqi;
		font: 500 4.6cqi / 1.2 var(--display);
		opacity: 0.85;
	}
	.light .beams {
		margin: 6cqi -12cqi;
		background: repeating-linear-gradient(
			-35deg,
			transparent 0 9cqi,
			rgb(255 255 255 / 0.85) 9cqi 9.6cqi,
			transparent 9.6cqi 16cqi
		);
		mask-image: linear-gradient(180deg, transparent, #000 30%, #000 70%, transparent);
	}
	.light .credit {
		display: flex;
		justify-content: space-between;
		font: 500 2.8cqi / 1.3 var(--font-sans);
		opacity: 0.7;
	}

	/* a quiet classic */
	.classic {
		grid-template-rows: 1fr auto auto;
		gap: 4%;
		background: var(--beige-50);
		color: var(--ink);
	}
	.photo.tall {
		margin: -8.5% -8.5% 0;
		background: var(--mist); /* the light hero blue */
	}
	.classic .t {
		font: 700 14cqi / 0.92 var(--display);
		letter-spacing: -0.02em;
	}
	.classic .s {
		justify-self: end;
		font: 500 5cqi / 1 var(--display);
	}

	/* text poster on sage */
	.sage {
		grid-template-rows: auto 1fr auto;
		background: var(--sage);
		color: var(--ink);
	}
	.sage .label {
		font: 600 3.4cqi / 1.2 var(--font-sans);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.sage .body {
		align-self: center;
		font: 600 6.4cqi / 1.15 var(--display);
		text-transform: uppercase;
		letter-spacing: 0.01em;
	}
	.sage .end {
		justify-self: end;
	}
</style>
