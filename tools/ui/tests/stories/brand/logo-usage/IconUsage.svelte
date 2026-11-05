<!--
	How the Llama icon is used, one case per section: as a glyph of Llama Sans (U+E000) in display type, in buttons,
	in a download badge and in running text, then as the web app's favicon, PWA, maskable and Apple touch icons and
	its splash screens, with their proportions and placement drawn in. The app icons follow the web UI's own
	pwa-assets.config.ts (PWA_ASSET_GENERATOR) and make-icons-circular.js, but are drawn with the L: the files it ships
	still carry the old llama.cpp mark. It sits in the logo showcase's frame (ShowcaseFrame), on its colours.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../logo-showcase/lib/logo';
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { FAVICON_COLORS, PWA_ASSET_GENERATOR, THEME_COLORS } from '$lib/constants';

	// @vite-pwa/assets-generator's default padding for the maskable and Apple touch icons
	const GENERATOR_ICON_PADDING = 0.3;
	// make-icons-circular.js scales the maskable icon to 85% before cutting the circle
	const CIRCULAR_ICON_SCALE = 0.85;
	// the icon sits 1/60 of its height above centre in a square or a circle: its weight is low (the wide bottom bar),
	// so mathematically centred it reads as sitting low (as in the brand's -bg files and social avatars)
	const OPTICAL_LIFT = 1 / 60;

	// every output starts from favicon.svg, the icon already shrunk by the favicon padding
	const favicon = 1 - PWA_ASSET_GENERATOR.FAVICON_PADDING;
	const square = (1 - GENERATOR_ICON_PADDING) * favicon;
	const circle = CIRCULAR_ICON_SCALE * square;
	// the splash icon is fitted into this share of the screen's shorter side
	const splash = (1 - PWA_ASSET_GENERATOR.SPLASH_PADDING) * favicon;

	const percent = (f: number) => `${Math.round(f * 1000) / 10}%`;

	interface AppIcon {
		background: string | null; // null: transparent, shown on a checkerboard (a dark one for a light icon)
		dark?: boolean;
		files: string;
		guide?: 'apple' | 'safe';
		ink: string;
		label: string;
		round?: boolean;
		scale: number;
		spec: string[];
	}

	const icons: AppIcon[] = [
		{
			background: null,
			files: 'favicon.svg, favicon.ico',
			ink: FAVICON_COLORS.LIGHT,
			label: 'Favicon',
			scale: favicon,
			spec: [
				`icon ${percent(favicon)} of the canvas`,
				`${percent((1 - favicon) / 2)} margin all round`,
				'transparent, ink for light tabs'
			]
		},
		{
			background: null,
			dark: true,
			files: 'favicon-dark.svg, favicon-dark.ico',
			ink: FAVICON_COLORS.DARK,
			label: 'Favicon, dark',
			scale: favicon,
			spec: [
				`icon ${percent(favicon)} of the canvas`,
				'transparent, light for dark tabs',
				'picked by prefers-color-scheme'
			]
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'pwa-64x64, -192x192, -512x512.png',
			ink: FAVICON_COLORS.LIGHT,
			label: 'PWA',
			round: true,
			scale: circle,
			spec: [
				`icon ${percent(circle)} of the circle`,
				'the maskable icon at 85%, cut round',
				'white, transparent outside'
			]
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'maskable-icon-512x512.png',
			guide: 'safe',
			ink: FAVICON_COLORS.LIGHT,
			label: 'Maskable',
			scale: square,
			spec: [
				`icon ${percent(square)} of the square`,
				'inside the 80% safe circle',
				'launchers crop to any shape'
			]
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-touch-icon-180x180.png',
			guide: 'apple',
			ink: FAVICON_COLORS.LIGHT,
			label: 'Apple touch',
			scale: square,
			spec: [
				`icon ${percent(square)} of the square`,
				'square file, white',
				'iOS rounds the corners'
			]
		}
	];

	const splashes = [
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-splash-portrait-*.png',
			h: 2556,
			ink: FAVICON_COLORS.LIGHT,
			label: 'iPhone',
			w: 1179
		},
		{
			background: THEME_COLORS.BACKGROUND_DARK,
			files: 'apple-splash-portrait-dark-*.png',
			h: 2556,
			ink: FAVICON_COLORS.DARK,
			label: 'iPhone, dark',
			w: 1179
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-splash-landscape-*.png',
			h: 2048,
			ink: FAVICON_COLORS.LIGHT,
			label: 'iPad, landscape',
			w: 2732
		}
	];

	// the icon placed in a w × h box at `scale` of its shorter side, centred, lifted optically
	function placed(w: number, h: number, scale: number) {
		const side = Math.min(w, h) * scale;
		const x = (w - side) / 2;
		const y = (h - side) / 2 - side * OPTICAL_LIFT;

		return { side, transform: `translate(${x} ${y}) scale(${side / SIZE})`, x, y };
	}

	const badges = [
		{ name: 'Llama', platform: 'macOS', top: 'Download for' },
		{ name: 'llama.cpp', platform: 'GitHub', top: 'Get it on' }
	];
</script>

<!-- 1. in type -->
<p class="kicker">In Llama Sans · U+E000</p>

<h2>A glyph of the font</h2>

<p>
	The icon is a character of Llama Sans, so anything set in it can carry the mark without an image:
	it sits on the baseline, reaches exactly the cap height and takes the text's colour and weight.
</p>

<div class="display">
	<p class="lockup"> llama.cpp</p>

	<p class="lockup"> Llama</p>
</div>

<!-- 2. in buttons -->
<p class="kicker">In buttons</p>

<h2>Open in Llama</h2>

<p>
	As a glyph, the icon centres on the button's line box exactly like the label's capitals, with a
	word space after it, so it needs no icon slot and no nudging. These are the design system's own
	buttons.
</p>

<div class="buttons">
	<div class="row">
		<Button size="sm"><span class="glyph"></span> Open in Llama</Button>

		<Button><span class="glyph"></span> Open in Llama</Button>

		<Button size="lg"><span class="glyph"></span> Open in Llama</Button>
	</div>

	<div class="row">
		<Button variant="outline"><span class="glyph"></span> llama.app</Button>

		<Button variant="secondary"><span class="glyph"></span> Run with llama.cpp</Button>

		<Button variant="ghost"><span class="glyph"></span> llama.cpp</Button>

		<Button aria-label="Llama" size="icon" variant="outline"><span class="glyph"></span></Button>
	</div>
</div>

<!-- 3. a download badge -->
<p class="kicker">Download badge</p>

<h2>Get the app</h2>

<p>
	Our own equivalent of a store badge, for links to download Llama or get llama.cpp: a dark key with
	the icon at the height of its two lines and a small line over the name. 48 px high (40 for a
	compact one), radius 10, the icon 24 px, 14 from the left edge and 10 from the text.
</p>

<div class="badges">
	{#each badges as b (b.name)}
		{#each ['dark', 'light'] as const as tone (tone)}
			<a class="badge {tone}" href="#top" onclick={(e) => e.preventDefault()}>
				<span aria-hidden="true" class="badge-icon"></span>

				<span class="badge-text">
					<span class="badge-top">{b.top}</span>

					<span class="badge-name">{b.platform}</span>
				</span>
			</a>
		{/each}
	{/each}

	<a class="badge dark compact" href="#top" onclick={(e) => e.preventDefault()}>
		<span aria-hidden="true" class="badge-icon"></span>

		<span class="badge-text">
			<span class="badge-top">Download for</span>

			<span class="badge-name">macOS</span>
		</span>
	</a>

	<a class="badge light compact" href="#top" onclick={(e) => e.preventDefault()}>
		<Download aria-hidden="true" class="badge-lucide" />

		<span class="badge-text">
			<span class="badge-top">Get the model</span>

			<span class="badge-name">GGUF</span>
		</span>
	</a>
</div>

<!-- 4. in running text -->
<p class="kicker">In running text</p>

<h2>Inline, at any size</h2>

<p>
	In a sentence the icon reads as a capital: the same height, the same weight, on the same baseline.
	Use it before the name it stands for, never as a word on its own.
</p>

<div class="inline">
	<p style:font-size="14px">
		Models load with  llama.cpp and run locally in  Llama, on your own machine.
	</p>

	<p style:font-size="17px">
		Models load with  llama.cpp and run locally in  Llama, on your own machine.
	</p>

	<p class="big">Run it in  Llama.</p>
</div>

<!-- 5. app icons -->
<p class="kicker">App icons</p>

<h2>On the tab and the home screen</h2>

<p>
	The web app's favicon, PWA, maskable and Apple touch icons, with their proportions drawn in. All
	of them start from the favicon, the icon with a {percent(PWA_ASSET_GENERATOR.FAVICON_PADDING / 2)} margin;
	the square icons add the generator's 30% padding, and the PWA icon is the maskable one at 85%, cut round.
	The icon sits a little above centre ({Math.round(OPTICAL_LIFT * 1000) / 10}% of its height), so it
	looks centred. The files the web UI ships still carry the old llama.cpp mark; these are drawn with
	the L.
</p>

<div class="icons">
	{#each icons as icon (icon.label)}
		{@const p = placed(100, 100, icon.scale)}
		<figure class="icon-card">
			<svg
				class:checker={!icon.background}
				class:checker-dark={!icon.background && icon.dark}
				class="icon-art"
				viewBox="0 0 100 100"
			>
				{#if icon.background}
					{#if icon.round}
						<circle cx="50" cy="50" fill={icon.background} r="50" />
					{:else}
						<rect fill={icon.background} height="100" width="100" />
					{/if}
				{/if}

				<g fill={icon.ink} transform={p.transform}>
					{#each PATHS as d (d)}<path {d} />{/each}
				</g>

				<!-- the icon's box -->
				<rect class="guide" height={p.side} width={p.side} x={p.x} y={p.y} />

				{#if icon.guide === 'safe'}
					<circle class="guide safe" cx="50" cy="50" r="40" />
				{:else if icon.guide === 'apple'}
					<rect class="guide safe" height="100" rx="22.5" width="100" />
				{/if}
			</svg>

			<figcaption>
				<span class="icon-label">{icon.label}</span>

				{#each icon.spec as line (line)}<span class="icon-spec">{line}</span>{/each}

				<span class="icon-files">{icon.files}</span>
			</figcaption>
		</figure>
	{/each}
</div>

<!-- 6. splash screens -->
<p class="kicker">Splash screens</p>

<h2>While the app opens</h2>

<p>
	For now the splash screens are the icon alone, centred on the light or dark page colour at
	{percent(splash)} of the screen's shorter side, for every iPhone and iPad size, upright and on its side.
	A designed splash screen will follow.
</p>

<div class="splashes">
	{#each splashes as s (s.label)}
		{@const p = placed(s.w, s.h, splash)}
		<figure class="splash-card">
			<svg style:aspect-ratio="{s.w} / {s.h}" class="splash-art" viewBox="0 0 {s.w} {s.h}">
				<rect fill={s.background} height={s.h} width={s.w} />

				<g fill={s.ink} transform={p.transform}>
					{#each PATHS as d (d)}<path {d} />{/each}
				</g>

				<rect class="guide" height={p.side} width={p.side} x={p.x} y={p.y} />
			</svg>

			<figcaption>
				<span class="icon-label">{s.label}</span>

				<span class="icon-spec">{s.w} × {s.h} · icon {Math.round(p.side)} px</span>

				<span class="icon-files">{s.files}</span>
			</figcaption>
		</figure>
	{/each}
</div>

<style>
	.glyph {
		font-family: 'Llama Sans', var(--body);
	}

	/* 1 */
	.display {
		display: grid;
		gap: 4px;
	}
	.lockup {
		margin: 0;
		font:
			700 clamp(48px, 9vw, 112px) / 1.15 'Llama Sans',
			var(--body);
	}

	/* 2 */
	.buttons {
		display: grid;
		gap: 16px;
		padding: clamp(20px, 3vw, 32px);
		border-radius: 16px;
		background: var(--l-panel);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px;
	}

	/* 3 */
	.badges {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px;
		padding: clamp(20px, 3vw, 32px);
		border-radius: 16px;
		background: var(--l-panel);
	}
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		height: 48px;
		padding: 0 18px 0 14px;
		border-radius: 10px;
		text-decoration: none;
	}
	.badge.dark {
		background: #111111;
		color: #fafafa;
		box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.14);
	}
	.badge.light {
		background: #fafafa;
		color: #111111;
		box-shadow: inset 0 0 0 1px rgb(17 17 17 / 0.16);
	}
	.badge-icon {
		font:
			700 24px / 1 'Llama Sans',
			var(--body);
	}
	.badge :global(.badge-lucide) {
		width: 22px;
		height: 22px;
	}
	.badge-text {
		display: grid;
		text-align: left;
	}
	.badge-top {
		font:
			500 10px / 1.2 'Llama Sans',
			var(--body);
		letter-spacing: 0.02em;
		opacity: 0.75;
	}
	.badge-name {
		font:
			600 18px / 1.15 'Llama Sans',
			var(--body);
		letter-spacing: -0.01em;
	}
	.badge.compact {
		height: 40px;
		padding: 0 15px 0 12px;
		gap: 8px;
		border-radius: 8px;
	}
	.compact .badge-icon {
		font-size: 20px;
	}
	.compact .badge-top {
		font-size: 9px;
	}
	.compact .badge-name {
		font-size: 15px;
	}

	/* 4 */
	.inline {
		display: grid;
		gap: 12px;
	}
	.inline p {
		margin: 0;
		max-width: 62ch;
		font-family: 'Llama Sans', var(--body);
		line-height: 1.5;
	}
	.inline .big {
		font:
			700 clamp(28px, 4vw, 44px) / 1.15 'Llama Sans',
			var(--body);
	}

	/* 5 and 6 */
	.icons {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 180px), 1fr));
		gap: 20px;
	}
	.splashes {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: 24px;
	}
	figure {
		display: grid;
		gap: 10px;
		align-content: start;
		margin: 0;
	}
	.icon-art {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		border-radius: 12px;
		background: var(--l-panel);
	}
	.icon-art.checker {
		/* fixed, like the tab it shows on: light for the dark icon, dark for the light one */
		background: repeating-conic-gradient(#ebebeb 0 25%, #fafafa 0 50%) 0 0 / 12px 12px;
	}
	.icon-art.checker-dark {
		background: repeating-conic-gradient(#1f1f1f 0 25%, #111111 0 50%) 0 0 / 12px 12px;
	}
	.splash-art {
		display: block;
		height: clamp(180px, 30vw, 280px);
		border-radius: 12px;
		box-shadow: 0 0 0 1px var(--l-rule);
	}
	.guide {
		fill: none;
		stroke: var(--l-mark);
		stroke-width: 1px;
		stroke-dasharray: 3 3;
		vector-effect: non-scaling-stroke;
	}
	.guide.safe {
		stroke: var(--l-muted);
	}
	figcaption {
		display: grid;
		gap: 2px;
	}
	.icon-label {
		font:
			600 15px / 1.3 'Llama Sans',
			var(--body);
	}
	.icon-spec {
		font:
			400 13px / 1.4 'Llama Sans',
			var(--body);
		color: var(--l-muted);
	}
	.icon-files {
		margin-top: 2px;
		font: 500 11px / 1.4 var(--mono);
		color: var(--l-muted);
		overflow-wrap: anywhere;
	}
</style>
