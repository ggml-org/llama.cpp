<!--
	How the Llama icon is used, one case per section (section picks which; each is a story of Brand/Logo/Usage, but
	the splash screens, which are UI/Foundations/Brand): as its two glyphs of Llama Sans (U+E000 for text, U+E001, the
	capital form at the lockup's proportion) in display type and buttons, the two lockups in context (Llama in Sans,
	llama.cpp in Mono), in a download badge and in running text, then as the web app's favicon, PWA, maskable and Apple
	touch icons and its splash screens, with their proportions and placement drawn in, the native apps' icons as they
	ship (from ggml-org/Llama-macOS and Llama-Windows), and in the social avatars, with how it is optically centred in
	a circle. The app icons follow the web UI's own pwa-assets.config.ts (PWA_ASSET_GENERATOR) and
	make-icons-circular.js, but are drawn with the L: the files it ships still carry the old llama.cpp mark. It sits
	in the logo showcase's frame (ShowcaseFrame), on its colours.
-->
<script lang="ts">
	import {
		MAC_DOCK_PLACEHOLDERS,
		MacDock,
		MacMenuBar,
		MockDesktop,
		WINDOWS_TASKBAR_PLACEHOLDERS,
		WindowsTaskbar
	} from '../../../../.storybook/mockups';
	import avatarSvg from '../assets/files/logo/svg/social/x/x-avatar-light.svg?raw';
	import macosIcon from '../assets/files/native/macos/icon_512x512@2x.png';
	import windowsIcon from '../assets/files/native/windows/Square44x44Logo.scale-400.png';
	import windowsTile from '../assets/files/native/windows/Square150x150Logo.scale-400.png';
	import OpticalCentre from '../assets/OpticalCentre.svelte';
	import { PATHS, SIZE } from '../logo-showcase/lib/logo';
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { FAVICON_COLORS, PWA_ASSET_GENERATOR, THEME_COLORS } from '$lib/constants';

	// @vite-pwa/assets-generator's default padding for the maskable and Apple touch icons
	const GENERATOR_ICON_PADDING = 0.3;
	// make-icons-circular.js scales the maskable icon to 85% before cutting the circle
	const CIRCULAR_ICON_SCALE = 0.85;
	// the icon sits 1/40 of its height above centre and 1/100 of its width left of it, in a square or a circle: its
	// weight is low (the wide bottom bar), so mathematically centred it reads as sitting low, and so placed a circle
	// round it clears its rounded corners, sharp corners and middle tips nearly evenly (as in the brand's -bg files,
	// social avatars and app icons)
	const OPTICAL_LIFT = 1 / 40;
	const OPTICAL_NUDGE = 1 / 100;

	// every output starts from favicon.svg, the icon already shrunk by the favicon padding
	const favicon = 1 - PWA_ASSET_GENERATOR.FAVICON_PADDING;
	const square = (1 - GENERATOR_ICON_PADDING) * favicon;
	const circle = CIRCULAR_ICON_SCALE * square;
	// the splash icon is fitted into this share of the screen's shorter side
	const splash = (1 - PWA_ASSET_GENERATOR.SPLASH_PADDING) * favicon;

	const percent = (f: number) => `${Math.round(f * 1000) / 10}%`;

	// the social avatar's icon placement, read from its file: its side, and the icon's translate and scale
	const avatar = (() => {
		const [, x, y, scale] = avatarSvg
			.match(/translate\(([\d.]+) ([\d.]+)\) scale\(([\d.]+)\)/)!
			.map(Number);

		return { scale, size: Number(avatarSvg.match(/width="(\d+)"/)![1]), x, y };
	})();

	// the native apps' icons as they ship: the macOS one hand-made as an appiconset, the Windows ones as the
	// package's assets
	const nativeIcons = [
		{
			files: 'AppIcon.appiconset, 16 to 1024',
			href: 'https://github.com/ggml-org/Llama-macOS/tree/main/Llama/Assets.xcassets/AppIcon.appiconset',
			label: 'macOS',
			spec: ['white icon on a #3F4040 squircle', 'Apple’s 824 plate on 1024'],
			src: macosIcon
		},
		{
			files: 'Square44x44Logo.*.png',
			href: 'https://github.com/ggml-org/Llama-Windows/tree/main/Assets',
			label: 'Windows, taskbar',
			spec: ['white icon on #111111', 'icon 66% of the square'],
			src: windowsIcon
		},
		{
			files: 'Square150x150Logo.*.png',
			href: 'https://github.com/ggml-org/Llama-Windows/tree/main/Assets',
			label: 'Windows, Start tile',
			spec: ['white icon on #111111', 'with a wide 310 × 150 tile'],
			src: windowsTile
		}
	];

	// the native apps where their OS puts them: Llama in the Dock (running, its name label showing), its status item in
	// the menu bar, and on the Windows taskbar (active) and in its tray
	const dockApps = [
		...MAC_DOCK_PLACEHOLDERS.slice(0, 3),
		{ icon: macosIcon, name: 'Llama', running: true },
		...MAC_DOCK_PLACEHOLDERS.slice(3)
	];
	const taskbarApps = [
		...WINDOWS_TASKBAR_PLACEHOLDERS.slice(0, 2),
		{ active: true, icon: windowsIcon, name: 'Llama', running: true },
		...WINDOWS_TASKBAR_PLACEHOLDERS.slice(2)
	];
	// Windows' tray icon: the icon alone, as llama.ico and llama-white.ico are, in the taskbar's ink
	const trayIcon = { paths: PATHS, viewBox: `0 0 ${SIZE} ${SIZE}` };

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

	// the icon placed in a w × h box at `scale` of its shorter side, centred, lifted and nudged optically
	function placed(w: number, h: number, scale: number) {
		const side = Math.min(w, h) * scale;
		const x = (w - side) / 2 - side * OPTICAL_NUDGE;
		const y = (h - side) / 2 - side * OPTICAL_LIFT;

		return { side, transform: `translate(${x} ${y}) scale(${side / SIZE})`, x, y };
	}

	// llama.app's hero and its navy night version, as on the brand's boards
	const NAVY =
		'radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%), linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%)';
	const HERO =
		'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';

	const badges = [
		{ name: 'Llama', platform: 'macOS', top: 'Download for' },
		{ name: 'llama.cpp', platform: 'GitHub', top: 'Get it on' }
	];

	let {
		section
	}: {
		section: 'badge' | 'buttons' | 'context' | 'glyphs' | 'icons' | 'splash' | 'text';
	} = $props();
</script>

<!-- 1. in type -->
{#if section === 'glyphs'}
	<p class="kicker">In Llama Sans · U+E002, U+E001 and U+E000</p>

	<h2>Three glyphs of the font</h2>

	<p>
		The icon is a character of Llama Sans, in two sizes, so anything set in it can carry the mark
		without an image, in the text's colour and weight. U+E001 is the one to reach for: the icon 1.24
		times the cap height, centred on the capitals, with the lockup's gap after it. Use it in small
		text, in controls and before a name, where a cap-height mark would read as smaller than the
		letters. U+E000 sits on the baseline at exactly the cap height: keep it for headlines, where the
		icon should sit quietly among large capitals and not stand out. In all caps, the case feature
		swaps U+E000 for U+E001. U+E002 is the lockup: U+E001 with the space after it built in, typed
		straight before the name, in Llama Sans or in Llama Mono, which carries it too.
	</p>

	<div class="display">
		<p class="lockup mono">llama.cpp</p>

		<p class="lockup">Llama</p>
	</div>

	<div class="pair">
		<p>
			<span class="tag">U+E002 · the lockup</span><span class="small-sample">Llama</span>
		</p>

		<p>
			<span class="tag">U+E001 · small text, controls, names</span><span class="small-sample"
				>Models run locally in  Llama.</span
			>
		</p>

		<p>
			<span class="tag">U+E000 · headlines</span><span class="sample">Run it in  Llama</span>
		</p>
	</div>
{/if}

<!-- 2. in buttons -->
{#if section === 'buttons'}
	<p class="kicker">In buttons</p>

	<h2>Open in Llama</h2>

	<p>
		Buttons always take the capital glyph, U+E001: at a label's size the cap-height glyph reads as
		smaller than the letters beside it. As a glyph, the icon centres on the button's line box like
		the label's capitals, with a word space after it, so it needs no icon slot and no nudging. These
		are the design system's own buttons.
	</p>

	<div class="buttons">
		<div class="row">
			<Button size="sm"><span class="glyph"></span> Open in Llama</Button>

			<Button><span class="glyph"></span> Open in Llama</Button>

			<Button size="lg"><span class="glyph"></span> Open in Llama</Button>
		</div>

		<div class="row">
			<Button variant="outline"><span class="glyph"></span> llama.app</Button>

			<Button variant="secondary"><span class="glyph"></span> Run with llama.cpp</Button>

			<Button variant="ghost"><span class="glyph"></span> llama.cpp</Button>

			<Button aria-label="Llama" size="icon" variant="outline"><span class="glyph"></span></Button>
		</div>
	</div>
{/if}

<!-- 3. the lockups: Llama in Llama Sans, llama.cpp in Llama Mono -->
{#if section === 'context'}
	<p class="kicker">The lockups in context</p>

	<h2>Where the name goes</h2>

	<p>
		The two lockups at the sizes they are used at: the README banner, app headers, a terminal, a
		footer and a hero. Llama, the app, is set in Llama Sans with its capital L; llama.cpp, the
		engine, in Llama Mono and all lowercase, as it is typed in a terminal. Both start with the
		lockup glyph, U+E002: the capital icon with the gap built in, so the lockup is one character and
		the name. Llama Mono's is drawn a little larger: with the icons matched, llama.cpp's wider,
		heavier letters set at 90% of Llama's. The llama.cpp lockup always sits on a neutral ground
		(white, the greys, ink); only the Llama lockup goes on colour, the hero's light blue and its
		navy.
	</p>

	<div class="contexts">
		<!-- the README banner, as on github.com/ggml-org/llama.cpp: the cover, 3:1, the lockup centred at its size -->
		{#each ['dark', 'light'] as const as tone (tone)}
			<div class="ctx banner mono {tone}">
				<span class="mark">llama<span class="cpp">.cpp</span></span>
			</div>
		{/each}

		<div class="ctx header mono">
			<span style:font-size="16.2px" class="mark">llama<span class="cpp">.cpp</span></span>

			<span class="links">docs · server · cli</span>

			<Button size="sm">Download</Button>
		</div>

		<div class="ctx header dark">
			<span style:font-size="16px" class="mark">Llama</span>

			<span class="links">Settings</span>
		</div>

		<!-- terminal output: Llama Mono's own one-cell icon, so the columns line up -->
		<pre class="ctx terminal"><span class="prompt">$</span> llama-server -m model.gguf --port 8080
<span class="banner"> llama.cpp</span> b6123 · listening on http://127.0.0.1:8080
<span class="banner"> llama.cpp</span> b6123 · slot 0 ready, ctx 32768</pre>

		<div class="ctx footer mono">
			<span style:font-size="11.7px" class="mark">llama<span class="cpp">.cpp</span></span>

			<span class="links">MIT licence · ggml-org</span>
		</div>

		<!-- the hero, on llama.app's light blue and its navy: the Llama lockup (llama.cpp's stays on neutral grounds) -->
		{#each [{ field: HERO, ink: '#111111', name: 'light' }, { field: NAVY, ink: '#fafafa', name: 'navy' }] as b (b.name)}
			<div style:background={b.field} style:color={b.ink} class="ctx hero">
				<span style:font-size="clamp(40px, 7vw, 64px)" class="mark">Llama</span>
			</div>
		{/each}
	</div>
{/if}

<!-- 4. a download badge -->
{#if section === 'badge'}
	<p class="kicker">Download badge</p>

	<h2>Get the app</h2>

	<p>
		Our own equivalent of a store badge, for links to download Llama or get llama.cpp: a dark key
		with the icon at the height of its two lines and a small line over the name. 48 px high (40 for
		a compact one), radius 10, the icon the capital glyph (U+E001) at 18 px, 14 from the left edge
		and 10 from the text.
	</p>

	<div class="badges">
		{#each badges as b (b.name)}
			{#each ['dark', 'light'] as const as tone (tone)}
				<a class="badge {tone}" href="#top" onclick={(e) => e.preventDefault()}>
					<span aria-hidden="true" class="badge-icon"></span>

					<span class="badge-text">
						<span class="badge-top">{b.top}</span>

						<span class="badge-name">{b.platform}</span>
					</span>
				</a>
			{/each}
		{/each}

		<a class="badge dark compact" href="#top" onclick={(e) => e.preventDefault()}>
			<span aria-hidden="true" class="badge-icon"></span>

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
{/if}

<!-- 5. in running text -->
{#if section === 'text'}
	<p class="kicker">In running text</p>

	<h2>Inline, at any size</h2>

	<p>
		In small text the icon takes its capital form, U+E001, so it holds its own next to the letters;
		in a headline, the cap-height U+E000 sits quietly among the capitals. Use it before the name it
		stands for, never as a word on its own.
	</p>

	<div class="inline">
		<p style:font-size="14px">
			Models load with  llama.cpp and run locally in  Llama, on your own machine.
		</p>

		<p style:font-size="17px">
			Models load with  llama.cpp and run locally in  Llama, on your own machine.
		</p>

		<p class="big">Run it in  Llama.</p>
	</div>
{/if}

<!-- 6. app icons -->
{#if section === 'icons'}
	<p class="kicker">App icons and avatars</p>

	<h2>On the tab, the home screen and the profile</h2>

	<!-- first how the icon is placed, optically centred, which every icon below is built on -->
	<p>
		Every icon starts from the same placement. Centred exactly, the L reads low, its weight in the
		wide foot; so the icon is lifted {Math.round(OPTICAL_LIFT * 1000) / 10}% of its height and
		nudged
		{Math.round(OPTICAL_NUDGE * 1000) / 10}% of its width left, and looks centred. Cut to a circle,
		as in the social avatars and the PWA icon, it then clears the circle's edge at its corners and
		tips nearly evenly. Drawn from the avatar file itself, the icon at 56% of a square, which every
		platform crops round.
	</p>

	<div class="optical">
		<OpticalCentre placement={avatar} />
	</div>

	<p class="icons-intro">
		The web app's favicon, PWA, maskable and Apple touch icons, with their proportions drawn in. All
		of them start from the favicon, the icon with a {percent(
			PWA_ASSET_GENERATOR.FAVICON_PADDING / 2
		)} margin; the square icons add the generator's 30% padding, and the PWA icon is the maskable one
		at 85%, cut round. The files the web UI ships still carry the old llama.cpp mark; these are drawn
		with the L.
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

	<!-- the native apps' icons as they ship, from ggml-org/Llama-macOS and ggml-org/Llama-Windows -->
	<p class="icon-label native-title">Native app icons</p>

	<div class="icons">
		{#each nativeIcons as icon (icon.label)}
			<figure class="icon-card">
				<img alt={icon.label} class="icon-art native-art" src={icon.src} />

				<figcaption>
					<a class="icon-label" href={icon.href}>{icon.label}</a>

					{#each icon.spec as line (line)}<span class="icon-spec">{line}</span>{/each}

					<span class="icon-files">{icon.files}</span>
				</figcaption>
			</figure>
		{/each}
	</div>

	<!-- the native apps in place, light and dark: a Mac's desktop, the menu bar over the Dock, and a PC's taskbar -->
	<div class="in-place">
		{#each ['light', 'dark'] as const as mode (mode)}
			<figure class="in-place-card">
				<MockDesktop align="between" height={250} {mode}>
					<MacMenuBar {mode} />

					<MacDock apps={dockApps} labelled="Llama" {mode} />
				</MockDesktop>

				<figcaption class="icon-label">macOS, the menu bar and the Dock, {mode}</figcaption>
			</figure>

			<figure class="in-place-card">
				<MockDesktop align="bottom" height={170} {mode}>
					<WindowsTaskbar apps={taskbarApps} {mode} {trayIcon} />
				</MockDesktop>

				<figcaption class="icon-label">Windows, the taskbar and the tray, {mode}</figcaption>
			</figure>
		{/each}
	</div>
{/if}

<!-- 7. splash screens -->
{#if section === 'splash'}
	<p class="kicker">Splash screens</p>

	<h2>While the app opens</h2>

	<p>
		For now the splash screens are the icon alone, centred on the light or dark page colour at
		{percent(splash)} of the screen's shorter side, for every iPhone and iPad size, upright and on its
		side. A designed splash screen will follow.
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
{/if}

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
	/* llama.cpp's U+E002 is 1 / 0.9 times Llama Sans's: at 90% of the size the icon matches Llama's, the name smaller */
	.lockup.mono {
		font-family: var(--mono);
		font-size: clamp(43px, 8.1vw, 101px);
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

	.pair {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 40px;
	}
	.pair p {
		display: grid;
		gap: 4px;
		margin: 0;
	}
	.tag {
		margin: 0;
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--l-muted);
	}
	.small-sample {
		font:
			400 15px / 1.5 'Llama Sans',
			var(--body);
	}
	.sample {
		font:
			700 44px / 1.15 'Llama Sans',
			var(--body);
	}

	/* 3 */
	.contexts {
		display: grid;
		gap: 12px;
	}
	.ctx {
		display: flex;
		align-items: center;
		gap: 16px;
		min-width: 0;
		padding: 0 16px;
		border-radius: 14px;
		background: var(--l-panel);
		color: var(--l-ink);
		overflow: hidden;
	}
	.ctx .mark {
		flex: none;
		font-family: 'Llama Sans', var(--body);
		font-weight: 700;
		line-height: 1;
		white-space: nowrap;
	}
	.ctx .cpp {
		color: var(--art-cinder);
	}
	.ctx .links {
		margin-left: auto;
		font:
			400 13px / 1.25 'Llama Sans',
			var(--body);
		color: var(--l-muted);
		white-space: nowrap;
	}
	.ctx.header {
		height: 56px;
		container-type: inline-size;
	}
	/* in a narrow header the links step aside, as they would in the app */
	@container (max-width: 26rem) {
		.ctx.header .links {
			display: none;
		}
		.ctx.header :global(button) {
			margin-left: auto;
		}
	}
	.ctx.dark {
		background: #111111;
		color: #fafafa;
	}
	.ctx.dark .links {
		color: rgb(250 250 250 / 0.6);
	}
	.ctx.footer {
		height: 64px;
	}
	.ctx.hero {
		justify-content: center;
	}
	.ctx.hero {
		min-height: 220px;
	}
	/* the cover's proportions: 3:1, the icon 20.8% of its height (520 of 2500), so the type is 7.1% of its width */
	.ctx.banner {
		justify-content: center;
		aspect-ratio: 3;
		container-type: inline-size;
		border-radius: 0;
	}
	.ctx.banner .mark {
		font-size: 7.1cqw;
	}
	.ctx.banner.dark {
		background: #111111;
		color: #fafafa;
	}
	.ctx.banner.light {
		background: #fafafa;
		color: #111111;
		box-shadow: inset 0 0 0 1px var(--l-rule);
	}

	/* llama.cpp's lockup in Llama Mono, which carries the same U+E002 */
	.ctx.mono .mark,
	.ctx.mono .links {
		font-family: var(--mono);
	}
	.ctx.terminal {
		display: block;
		margin: 0;
		padding: 16px;
		background: #111111;
		color: #fafafa;
		font: 400 13px / 1.6 var(--mono);
		overflow-x: auto;
	}
	.terminal .prompt {
		color: rgb(250 250 250 / 0.5);
	}
	.terminal .banner {
		color: var(--art-cinder);
		font-weight: 700;
	}

	/* 4 */
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
	/* the capital glyph at 20 px: an 18 px icon */
	.badge-icon {
		font:
			700 20px / 1 'Llama Sans',
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
		font-size: 17px;
	}
	.compact .badge-top {
		font-size: 9px;
	}
	.compact .badge-name {
		font-size: 15px;
	}

	/* 5 */
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

	/* 6 and 7 */
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
	.optical {
		max-width: 54rem;
		margin-top: 2rem;
	}
	/* the icons' lead, after the placement, set like the section's own under its heading */
	.icons-intro {
		max-width: 62ch;
		margin-top: 2.5rem;
		color: var(--l-muted);
		font:
			400 17px / 1.5 'Llama Sans',
			var(--body);
	}
	/* the native apps' icons: their files as they are, on their own grounds */
	.native-title {
		margin-top: 2.5rem;
		margin-bottom: -20px;
	}
	.native-art {
		border-radius: 0;
		background: transparent;
		object-fit: contain;
	}
	/* the native apps in place, each desktop across */
	.in-place {
		display: grid;
		gap: 24px;
	}
	.in-place-card {
		display: grid;
		gap: 8px;
		min-width: 0;
		margin: 0;
	}
</style>
