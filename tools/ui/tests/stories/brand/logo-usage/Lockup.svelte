<!--
	The two lockups, Llama (Llama Sans) and llama.cpp (Llama Mono), on one page: how each is built, drawn over the
	brand's own files (the icon, the gap U+E002 builds in, the cap height or ascenders it centres on, the baseline,
	and the clear space round it), how each is typed, the sizes they are set at, the grounds each goes on, what they
	never do, and the files. The lockups in context follow, from IconUsage. It sits in the logo showcase's frame
	(ShowcaseFrame), on its colours.
-->
<script lang="ts">
	import llamaDark from '../assets/files/logo/svg/lockup/llama/logo-llama-dark.svg?url';
	import llamaDarkBg from '../assets/files/logo/svg/lockup/llama/logo-llama-dark-bg.svg?url';
	import llamaRaw from '../assets/files/logo/svg/lockup/llama/logo-llama-light.svg?raw';
	import llamaLight from '../assets/files/logo/svg/lockup/llama/logo-llama-light.svg?url';
	import llamaLightBg from '../assets/files/logo/svg/lockup/llama/logo-llama-light-bg.svg?url';
	import cppDark from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-dark.svg?url';
	import cppDarkBg from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-dark-bg.svg?url';
	import cppRaw from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light.svg?raw';
	import cppLight from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light.svg?url';
	import cppLightBg from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light-bg.svg?url';
	import IconUsage from './IconUsage.svelte';

	// the files' drawing, in their own units (600 high), with the ink swapped for the text colour so it follows the
	// theme; llama.cpp's .cpp keeps its cinder
	const art = (raw: string) =>
		raw
			.replace(/^[\s\S]*?<svg[^>]*>/, '')
			.replace(/<\/svg>\s*$/, '')
			.replaceAll('#111111', 'currentColor');

	// measured from the files: the icon is 520 high from y 40, and the name starts after the gap U+E002 builds in
	const ICON = { bottom: 560, side: 520, top: 40 };
	// the clear space all round: half the icon
	const CLEAR = ICON.side / 2;

	const lockups = [
		{
			art: art(llamaRaw),
			base: 509.8,
			end: 2317,
			face: 'Llama Sans',
			line: 'cap height',
			name: 'Llama',
			start: 754.6,
			top: 90.2,
			width: 2342
		},
		{
			art: art(cppRaw),
			base: 488.8,
			end: 3584.2,
			face: 'Llama Mono',
			line: 'ascenders',
			name: 'llama.cpp',
			start: 763,
			top: 111.2,
			width: 3607
		}
	];

	const files = [
		{ ground: 'light', name: 'logo-llama-light-bg', src: llamaLightBg },
		{ ground: 'dark', name: 'logo-llama-dark-bg', src: llamaDarkBg },
		{ ground: 'light', name: 'logo-llama-light', src: llamaLight },
		{ ground: 'dark', name: 'logo-llama-dark', src: llamaDark },
		{ ground: 'light', name: 'logo-llama-cpp-light-bg', src: cppLightBg },
		{ ground: 'dark', name: 'logo-llama-cpp-dark-bg', src: cppDarkBg },
		{ ground: 'light', name: 'logo-llama-cpp-light', src: cppLight },
		{ ground: 'dark', name: 'logo-llama-cpp-dark', src: cppDark }
	];

	const sizes = [64, 32, 16, 12];

	// llama.app's hero, the day sky and the night sky, as on the brand's boards
	const NIGHT_SKY =
		'radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%), linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%)';
	const DAY_SKY =
		'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';

	const grounds = [
		{ field: '#fafafa', ink: '#111111', name: 'White' },
		{ field: '#111111', ink: '#fafafa', name: 'Ink' },
		{ field: DAY_SKY, ink: '#111111', name: 'Day sky', sky: true },
		{ field: NIGHT_SKY, ink: '#fafafa', name: 'Night sky', sky: true }
	];
</script>

{#snippet llama()}<span class="mark sans">&#xE002;Llama</span>{/snippet}

{#snippet cpp()}<span class="mark mono">&#xE002;llama<span class="cpp">.cpp</span></span>{/snippet}

<p class="kicker">The lockups · Llama and llama.cpp</p>

<h2>The icon and the name</h2>

<p>
	There are two lockups: Llama, the app, in Llama Sans, and llama.cpp, the engine, in Llama Mono,
	lowercase as it is typed in a terminal. Both are the icon and the name as one piece, built the
	same way, so either can be typed in its font as well as placed from a file.
</p>

<!-- 1. construction, drawn over the files -->
<section class="block">
	<h3>How they are built</h3>

	<p class="lead">
		The icon is 1.24 times Llama's cap height and centred on the capitals; in llama.cpp it is
		centred on the ascenders. The gap after it is part of the glyph, U+E002, so it never needs
		measuring. Keep half the icon's height clear all round, and nothing else inside that space.
	</p>

	{#each lockups as l (l.name)}
		{@const w = l.end + 2 * CLEAR}
		<figure class="construction">
			<svg
				aria-label="How the {l.name} lockup is built"
				role="img"
				viewBox="{-CLEAR - 40} {ICON.top - CLEAR - 40} {w + 80} {ICON.side + 2 * CLEAR + 80}"
			>
				<!-- the clear space, half the icon all round -->
				<rect
					class="clear"
					height={ICON.side + 2 * CLEAR}
					width={l.end + 2 * CLEAR}
					x={-CLEAR}
					y={ICON.top - CLEAR}
				/>

				<rect class="ground" height={ICON.side} width={l.end} x="0" y={ICON.top} />

				<!-- one clear space unit in a corner, and the icon's box -->
				<rect class="unit" height={CLEAR} width={CLEAR} x={-CLEAR} y={ICON.top - CLEAR} />

				<rect class="box" height={ICON.side} width={ICON.side} x="0" y={ICON.top} />

				<!-- the line the icon centres on, and the baseline -->
				<line class="guide" x1={-CLEAR} x2={l.end + CLEAR} y1={l.top} y2={l.top} />

				<line class="guide base" x1={-CLEAR} x2={l.end + CLEAR} y1={l.base} y2={l.base} />

				<line class="guide mid" x1={-CLEAR} x2={l.end + CLEAR} y1="300" y2="300" />

				<!-- the gap U+E002 builds in -->
				<rect
					class="gap"
					height={ICON.side}
					width={l.start - ICON.side}
					x={ICON.side}
					y={ICON.top}
				/>

				<g class="art">{@html l.art}</g>

				<text x={-CLEAR + 16} y={ICON.top - CLEAR + 52}>½ icon</text>

				<text x={ICON.side + 16} y={ICON.bottom + 56}>gap</text>

				<text text-anchor="end" x={l.end + CLEAR - 16} y={l.top - 18}>{l.line}</text>

				<text text-anchor="end" x={l.end + CLEAR - 16} y={l.base + 56}>baseline</text>
			</svg>

			<figcaption>
				<b>{l.name}</b>

				<span>{l.face} Bold · U+E002, then the name · centred on the {l.line}</span>
			</figcaption>
		</figure>
	{/each}
</section>

<!-- 2. typed -->
<section class="block">
	<h3>Typed</h3>

	<p class="lead">
		Type U+E002 straight before the name, with no space, in Bold. Llama Mono's icon is drawn 1/0.9
		as large as Llama Sans's, so set llama.cpp at 90% of Llama's size and the two icons match.
	</p>

	<div class="typed">
		<p class="big">{@render llama()}</p>

		<p class="big cpp-size">{@render cpp()}</p>
	</div>

	<dl class="parts">
		<dt>U+E002</dt>

		<dd>The icon at 1.24 times the cap height, with the gap after it built in</dd>

		<dt>Llama</dt>

		<dd>Llama Sans Bold, with its capital L</dd>

		<dt>llama.cpp</dt>

		<dd>Llama Mono Bold, always lowercase; .cpp in cinder, #EE6C1E, the brand's one accent</dd>
	</dl>
</section>

<!-- 3. sizes -->
<section class="block">
	<h3>Sizes</h3>

	<p class="lead">
		From a banner down to 12 px, the smallest either is set at. Below that, use the icon alone.
	</p>

	<div class="sizes">
		{#each sizes as size (size)}
			<p>
				<span class="label">{size} px</span>

				<span style:font-size="{size}px">{@render llama()}</span>

				<span style:font-size="{size * 0.9}px">{@render cpp()}</span>
			</p>
		{/each}
	</div>
</section>

<!-- 4. grounds -->
<section class="block">
	<h3>Grounds</h3>

	<p class="lead">
		Both go on white, the greys and ink, in ink or white. Only Llama goes on colour, on the day sky
		and the night sky; llama.cpp stays on neutral grounds.
	</p>

	<div class="grounds">
		{#each grounds as g (g.name)}
			<figure style:background={g.field} style:color={g.ink} class="ground-card">
				<span class="pair">
					<span>{@render llama()}</span>

					{#if !g.sky}<span class="cpp-size">{@render cpp()}</span>{/if}
				</span>

				<figcaption>{g.name}{g.sky ? ' · Llama only' : ''}</figcaption>
			</figure>
		{/each}
	</div>
</section>

<!-- 5. never -->
<section class="block">
	<h3>Never</h3>

	<div class="donts">
		<figure>
			<p><span class="mark mono">&#xE002;Llama</span></p>

			<figcaption>Llama in Llama Mono: that face is llama.cpp's</figcaption>
		</figure>

		<figure>
			<p class="cpp-size">
				<span class="mark sans">&#xE002;llama<span class="cpp">.cpp</span></span>
			</p>

			<figcaption>llama.cpp in Llama Sans: that face is Llama's</figcaption>
		</figure>

		<figure>
			<p class="cpp-size">
				<span class="mark mono">&#xE002;LLAMA<span class="cpp">.CPP</span></span>
			</p>

			<figcaption>In capitals: llama.cpp is a command, typed as one</figcaption>
		</figure>

		<figure>
			<p><span class="mark sans">&#xE002;&nbsp;&nbsp;Llama</span></p>

			<figcaption>A space after the icon: the gap is already built in</figcaption>
		</figure>

		<figure>
			<p><span class="mark sans stretched">&#xE002;Llama</span></p>

			<figcaption>Stretched, squeezed or slanted</figcaption>
		</figure>

		<figure>
			<p><span class="mark sans"><span class="cpp">&#xE002;</span>Llama</span></p>

			<figcaption>Two colours: the Llama lockup is one colour</figcaption>
		</figure>

		<figure style:background={DAY_SKY} style:color="#111111">
			<p class="cpp-size">{@render cpp()}</p>

			<figcaption>llama.cpp on colour: it stays on neutral grounds</figcaption>
		</figure>

		<figure>
			<p><span class="mark sans shadow">&#xE002;Llama</span></p>

			<figcaption>Shadows, outlines, gradients or other effects</figcaption>
		</figure>
	</div>
</section>

<!-- 6. files -->
<section class="block">
	<h3>Files</h3>

	<p class="lead">
		Each lockup in ink for light grounds and in white for dark ones, alone or on its ground (-bg).
		They are in Brand › Logo › Assets.
	</p>

	<div class="files">
		{#each files as f (f.name)}
			<figure class="file {f.ground}">
				<img
					alt="The {f.name.includes('cpp') ? 'llama.cpp' : 'Llama'} lockup, {f.name}"
					src={f.src}
				/>

				<figcaption>{f.name}.svg</figcaption>
			</figure>
		{/each}
	</div>
</section>

<!-- 7. in context, as the usage pages show it -->
<IconUsage section="context" />

<style>
	h3 {
		margin: 0;
		font:
			600 20px / 1.3 'Llama Sans',
			var(--body);
	}
	.block {
		display: grid;
		gap: 16px;
	}
	.block > * {
		margin: 0;
		min-width: 0;
	}
	.lead {
		max-width: 62ch;
		color: var(--l-muted);
	}

	/* the lockups, typed: Llama in Llama Sans, llama.cpp in Llama Mono at 90% so the icons match */
	.mark {
		font-weight: 700;
		white-space: nowrap;
	}
	.sans {
		font-family: 'Llama Sans', var(--body);
	}
	.mono {
		font-family: var(--mono);
	}
	.cpp {
		color: #ee6c1e;
	}
	.cpp-size {
		font-size: 0.9em;
	}

	/* 1 */
	.construction {
		display: grid;
		gap: 10px;
		margin: 0;
		padding: clamp(16px, 3vw, 32px);
		border-radius: 14px;
		background: var(--l-panel);
		color: var(--l-ink);
	}
	.construction svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.clear {
		fill: none;
		stroke: var(--l-line);
		stroke-dasharray: 6 6;
		vector-effect: non-scaling-stroke;
	}
	.ground {
		fill: var(--l-grid);
	}
	.unit,
	.gap {
		fill: color-mix(in oklab, #ee6c1e 14%, transparent);
	}
	.box {
		fill: none;
		stroke: var(--l-line);
		vector-effect: non-scaling-stroke;
	}
	.guide {
		stroke: var(--l-line);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.guide.mid {
		stroke-dasharray: 2 4;
	}
	.guide.base {
		stroke: #ee6c1e;
	}
	.art {
		color: var(--l-ink);
	}
	text {
		fill: var(--l-muted);
		font:
			500 40px var(--mono),
			monospace;
	}
	.construction figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		align-items: baseline;
		font-size: 14px;
	}
	.construction figcaption span {
		color: var(--l-muted);
		font: 400 12px var(--mono);
	}

	/* 2 */
	.typed {
		display: grid;
		gap: 8px;
		padding: clamp(24px, 4vw, 48px);
		border-radius: 14px;
		background: var(--l-panel);
	}
	.big {
		margin: 0;
		font-size: clamp(40px, 8vw, 96px);
		line-height: 1.2;
	}
	.parts {
		display: grid;
		grid-template-columns: 7rem 1fr;
		gap: 8px 16px;
		margin: 0;
	}
	.parts dt {
		font: 600 14px / 1.6 var(--mono);
	}
	.parts dd {
		margin: 0;
		font-size: 15px;
	}

	/* 3 */
	.sizes {
		display: grid;
		gap: 12px;
		padding: clamp(20px, 3vw, 32px);
		border-radius: 14px;
		background: var(--l-panel);
	}
	.sizes p {
		display: grid;
		grid-template-columns: 4rem 1fr 1fr;
		align-items: center;
		gap: 16px;
		margin: 0;
		overflow: hidden;
	}
	.label {
		color: var(--l-muted);
		font: 500 12px var(--mono);
	}

	/* 4 */
	.grounds {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
		gap: 16px;
	}
	.ground-card {
		display: grid;
		gap: 24px;
		margin: 0;
		padding: 32px 20px 14px;
		border-radius: 14px;
		box-shadow: inset 0 0 0 1px var(--l-rule);
	}
	.pair {
		display: grid;
		justify-items: center;
		gap: 12px;
		font-size: 28px;
	}
	.ground-card figcaption {
		font: 500 12px var(--mono);
		opacity: 0.65;
	}

	/* 5 */
	.donts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
		gap: 16px;
	}
	.donts figure {
		position: relative;
		display: grid;
		gap: 12px;
		margin: 0;
		padding: 32px 20px 16px;
		border-radius: 14px;
		background: var(--l-panel);
	}
	/* a cinder cross in the corner */
	.donts figure::after {
		content: '';
		position: absolute;
		inset: 12px 12px auto auto;
		width: 18px;
		height: 18px;
		background: #ee6c1e;
		mask:
			linear-gradient(45deg, transparent 44%, #000 44% 56%, transparent 56%),
			linear-gradient(-45deg, transparent 44%, #000 44% 56%, transparent 56%);
	}
	.donts p {
		margin: 0;
		font-size: clamp(24px, 3vw, 30px);
		text-align: center;
	}
	.donts figcaption {
		color: var(--l-muted);
		font-size: 13px;
		line-height: 1.4;
	}
	.donts figure[style] figcaption {
		color: inherit;
		opacity: 0.7;
	}
	.stretched {
		display: inline-block;
		transform: scaleX(1.5);
	}
	.shadow {
		text-shadow: 0 3px 6px rgb(0 0 0 / 0.35);
		-webkit-text-stroke: 1px #ee6c1e;
	}

	/* 6 */
	.files {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
		gap: 16px;
	}
	.file {
		display: grid;
		margin: 0;
		overflow: hidden;
		border-radius: 14px;
	}
	.file.light {
		background: #fafafa;
		color: #111111;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.1);
	}
	.file.dark {
		background: #111111;
		color: #fafafa;
	}
	.file img {
		display: block;
		width: 100%;
		height: auto;
		padding: 14% 12%;
		box-sizing: border-box;
	}
	/* the -bg files bring their own ground */
	.file:nth-child(4n + 1) img,
	.file:nth-child(4n + 2) img {
		padding: 0;
	}
	.file figcaption {
		padding: 8px 12px;
		font: 500 12px var(--mono);
		opacity: 0.6;
	}
</style>
