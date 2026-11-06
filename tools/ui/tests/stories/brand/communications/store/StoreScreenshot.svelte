<!--
	A Microsoft Store screenshot: one practical point in a centred headline, and under it the thing that proves it,
	large and running off the bottom edge, as the Store's best listings do. `window` frames a real app scene (a story
	of Store scenes) in a Windows 11 window; `desktop` is Llama's flyout over the taskbar; `brand` is the opener, the
	headline at display size over the Llama lockup; `art` takes a figure of its own (a motif, a code card). The
	headline keeps to the top quarter; the Store's text over the bottom quarter only ever covers the picture.
-->
<script lang="ts">
	import { WindowsTaskbar } from '../../../../../.storybook/mockups';
	import { PATHS, SIZE } from '../../logo-showcase/lib/logo';
	import AppPanel from '../posts/AppPanel.svelte';
	import { FORMATS, grid, type Ground, ICON, LOCKUP } from '../posts/post';
	import PostFrame from '../posts/PostFrame.svelte';
	import { type Scene, SCENES } from './scenes';
	import type { Snippet } from 'svelte';

	interface Props {
		children?: Snippet;
		headline: string;
		kind: 'art' | 'brand' | 'desktop' | 'window';
		lede?: string;
		overlay?: boolean;
		/** the app scene a `window` shows */
		scene?: Scene;
		/** the window's title */
		title?: string;
		tone: Ground;
	}

	let {
		children,
		headline,
		kind,
		lede,
		overlay,
		scene = 'chat',
		title = 'Llama',
		tone
	}: Props = $props();

	const format = FORMATS.msScreenshot;
	const g = grid(format);
	// the window is 30 units wide; the desktop is drawn at 1.6 times Windows' size, its taskbar 48px high
	const windowZoom = (g.unit * 30) / 1040;
	const desk = 1.6;
	// from the area's bottom down to the piece's edge
	const below = g.margin + (format.safe?.bottom ?? 0);
	const trayIcon = { paths: PATHS, viewBox: `0 0 ${SIZE} ${SIZE}` };
</script>

<PostFrame {format} label="Store screenshot, {kind}, {tone}" {overlay} {tone}>
	<div class={['shot', `kind-${kind}`]}>
		<div class="text">
			<h2 class={kind === 'brand' ? 't-display' : 'headline'}>{headline}</h2>

			{#if lede}<p class="t-lede">{lede}</p>{/if}
		</div>

		{#if kind === 'window'}
			<!-- the window is laid out at 1040 px, a small laptop's, and zoomed to the piece so the app reads at Store size -->
			<div style:margin-bottom="{-below / windowZoom}px" style:zoom={windowZoom} class="window">
				<div class="titlebar">
					<span class="app-icon">{ICON}</span>

					<span class="title">{title}</span>

					<span aria-hidden="true" class="captions">
						<svg viewBox="0 0 46 32"><path d="M18 16h10" /></svg>

						<svg viewBox="0 0 46 32"><rect height="9" rx="1.5" width="9" x="18.5" y="11.5" /></svg>

						<svg viewBox="0 0 46 32"><path d="M18 11l10 10M28 11 18 21" /></svg>
					</span>
				</div>

				<iframe
					loading="eager"
					src="iframe.html?id={SCENES[scene]}&viewMode=story&globals=interactionPace:instant"
					tabindex="-1"
					title="Llama, {scene}"
				></iframe>
			</div>
		{:else if kind === 'desktop'}
			<!-- a flyout opens 12px over the taskbar -->
			<div style:bottom="{(48 + 12) * desk - below}px" class="flyout">
				<div style:zoom={desk}><AppPanel bar={false} platform="windows" /></div>
			</div>

			<div
				style:bottom="{-below}px"
				style:left="{-g.margin}px"
				style:width="{format.width}px"
				class="taskbar"
			>
				<div style:width="{format.width / desk}px" style:zoom={desk}>
					<WindowsTaskbar clock={['10:24', '06/11/2026']} {trayIcon} />
				</div>
			</div>
		{:else if kind === 'brand'}
			<span class="lockup">{LOCKUP}Llama</span>
		{:else if children}
			<div class="art">{@render children()}</div>
		{/if}
	</div>
</PostFrame>

<style>
	.shot {
		position: relative;
		display: flex;
		flex: 1;
		flex-direction: column;
		align-items: center;
		min-height: 0;
	}
	.text {
		display: grid;
		gap: calc(var(--u) * 0.5);
		justify-items: center;
		max-width: calc(var(--u) * 34);
		text-align: center;
	}
	.headline {
		margin: 0;
		font: 700 calc(var(--u) * 1.7) / 1.02 var(--sans);
		letter-spacing: -0.025em;
		white-space: pre-line;
		text-wrap: balance;
	}
	.text :global(.t-lede) {
		margin: 0;
		text-wrap: balance;
	}

	/*
		a window: 1040px wide, zoomed to 30 units, down to the piece's bottom edge, where it is cut off (the app gets the
		height that shows, so its own bottom, the message box, stays in view)
	*/
	.window {
		display: flex;
		flex: 1 1 0;
		flex-direction: column;
		width: 1040px;
		min-height: 0;
		margin-top: 40px;
		overflow: hidden;
		border: 1px solid rgb(17 17 17 / 0.12);
		border-bottom: 0;
		border-radius: 8px 8px 0 0;
		background: #fff;
		box-shadow:
			0 32px 64px -24px rgb(17 34 60 / 0.35),
			0 2px 8px rgb(17 34 60 / 0.08);
	}
	.titlebar {
		display: flex;
		align-items: center;
		gap: 10px;
		height: 32px;
		padding-left: 12px;
		background: #f3f3f3;
		color: #111;
		font: 400 12px / 1 var(--sans);
	}
	.app-icon {
		font: 700 14px / 1 var(--sans);
	}
	.title {
		flex: 1;
	}
	.captions {
		display: flex;
	}
	.captions svg {
		width: 46px;
		height: 32px;
		fill: none;
		stroke: #111;
		stroke-width: 1;
	}
	iframe {
		display: block;
		flex: 1;
		width: 1040px;
		border: 0;
		pointer-events: none;
	}

	/* the desktop: the flyout over the full taskbar, which runs edge to edge at the bottom */
	.flyout {
		position: absolute;
		right: 0;
	}
	.taskbar {
		position: absolute;
	}

	/* the opener: the statement big in the middle, the lockup under it */
	.kind-brand {
		justify-content: center;
		gap: calc(var(--u) * 2);
	}
	.kind-brand .t-display {
		white-space: pre-line;
	}
	.lockup {
		font: 700 calc(var(--u) * 1.4) / 1 var(--sans);
		letter-spacing: -0.01em;
	}

	.art {
		display: flex;
		flex: 1;
		align-items: center;
		justify-content: center;
		width: 100%;
		min-height: 0;
		margin-top: calc(var(--u) * 1);
		/* the picture may run into the bottom quarter; only the words keep out of it */
		margin-bottom: calc(-1 * var(--safe-bottom));
	}
</style>
