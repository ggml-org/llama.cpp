<!--
	Specimen boards and posters for Llama Sans, after the classic Swiss specimen boards, in the brand palette: four
	boards (specimen, title, alphabet, a wall of small posters), then three posters on the communications grounds
	(dark, sand under the night sky, the day sky), cinder once on each; no red.
	Light mixes stay within one hue. Fixed to light, so they read the same in both themes. Set in Llama Sans Display
	and UI.
-->
<script lang="ts">
	import '../fonts/fonts.css';

	const upper = 'ABCDEFGHIJKLMN\nOPQRSTUVWXYZ';
	const lower = 'abcdefghijklmn\nopqrstuvwxyz()&?!';
	const figures = '1234567890.,;:’”/–';
	const rows = [400, 500, 600, 700, 400, 500, 600, 700];
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

	<!-- an exhibition poster with light beams -->
	<article aria-label="Explorations in light" class="board poster light">
		<p class="t">Explorations in light</p>

		<p class="s">An exhibition of type</p>

		<div class="beams"></div>

		<p class="credit">
			<span>Set in Llama Sans<br />Display &amp; UI</span><span class="date">11<br />26</span>
		</p>
	</article>

	<!-- a quiet classic: a night sky over sand, under a short title -->
	<article aria-label="A quiet classic" class="board poster classic">
		<div class="photo sky tall"></div>

		<p class="t">Llama<br />Sans</p>

		<p class="s">A quiet classic</p>
	</article>

	<!-- a text poster on the day sky -->
	<article aria-label="About Llama Sans" class="board poster note">
		<p class="label"><i class="dot"></i>Llama Sans</p>

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
		/* every colour from the brand palette: the art colours (art-*), cinder, the greys, and sand as the warm
		   neutral; mixes stay within one hue */
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
		--mist: linear-gradient(
			160deg,
			color-mix(in oklab, var(--art-sky), white 55%) 0%,
			color-mix(in oklab, var(--art-sky), white 68%) 100%
		);
		--cream: var(--sand-100);
		/* the day and night skies, as the grounds */
		--day-sky:
			radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%),
			linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%);
		--night-sky:
			radial-gradient(ellipse 120% 60% at 50% 115%, rgb(46 70 104 / 0.35) 0%, transparent 70%),
			linear-gradient(180deg, #0f1a2b 0%, #13213a 55%, #182a45 100%);
		--cinder: var(--art-cinder); /* the accent orange and the .cpp of the logotype */
		--ink: var(--sand-900);
		--display: 'Llama Sans', var(--font-sans);
		display: grid;
		/* twelve columns: the boards take halves, the posters thirds */
		grid-template-columns: repeat(12, minmax(0, 1fr));
		gap: 14px;
	}
	@media (max-width: 900px) {
		.wide {
			grid-column: span 12 !important;
		}
		.poster {
			grid-column: span 6 !important;
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
		grid-column: span 6;
		aspect-ratio: 16 / 10;
		padding: 5%;
	}
	.poster {
		grid-column: span 4;
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
		background: var(--sand-200);
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
		background: var(--sand-50);
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
		background: var(--cinder);
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
			color-mix(in oklab, var(--art-cinder), white 12%) 0%,
			var(--art-cinder) 100%
		);
	}
	.photo.mist {
		background: linear-gradient(
			180deg,
			color-mix(in oklab, var(--art-amber), white 12%) 0%,
			var(--art-amber) 100%
		); /* saffron */
	}

	/* exhibition poster */
	.light {
		grid-template-rows: auto auto 1fr auto;
		background: #111111; /* the dark ground */
		color: #fafafa;
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
		color: rgb(250 250 250 / 0.7);
	}
	.light .date {
		color: var(--cinder); /* the one accent */
	}

	/* a quiet classic */
	.classic {
		grid-template-rows: 1fr auto auto;
		gap: 4%;
		background: var(--sand-50);
		color: var(--ink);
	}
	.photo.tall {
		margin: -8.5% -8.5% 0;
		background: var(--night-sky); /* the night sky, over sand */
	}
	.classic .t {
		font: 700 14cqi / 0.92 var(--display);
		letter-spacing: -0.02em;
	}
	.classic .s {
		justify-self: end;
		font: 500 5cqi / 1 var(--display);
	}

	/* text poster on the day sky */
	.note {
		grid-template-rows: auto 1fr auto;
		background: var(--day-sky);
		color: #111111;
	}
	.note .label {
		display: flex;
		gap: 1.6cqi;
		align-items: center;
		font: 600 3.4cqi / 1.2 var(--font-sans);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.note .dot {
		width: 2.4cqi;
		height: 2.4cqi;
		border-radius: 50%;
		background: var(--cinder);
	}
	.note .body {
		align-self: center;
		font: 600 7cqi / 1.12 var(--display);
		letter-spacing: -0.01em;
		text-wrap: pretty;
	}
	.note .end {
		justify-self: end;
		color: rgb(17 17 17 / 0.62);
	}
</style>
