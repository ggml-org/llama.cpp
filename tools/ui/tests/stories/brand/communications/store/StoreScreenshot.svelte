<!--
	A Microsoft Store screenshot: one practical point in a centred headline, and under it the thing that proves it,
	large, as the Store's best listings do. Drawn after Llama for Windows as it is, in Windows' dark theme: `window` is
	its quick chat, a frameless window with the running model in a pill and Llama Chat inside (a real app scene, a
	story of Store scenes); `desktop` is its tray flyout over the taskbar, flush with the screen's edge; `card` is the
	flyout's Models card up close, running off the bottom; `brand` is the opener, the headline at display size over
	the Llama lockup; `art` takes a figure of its own (a motif, a code card). The headline keeps to the top quarter;
	the Store's text over the bottom quarter only ever covers the picture.
-->
<script lang="ts">
	import { WindowsTaskbar } from '../../../../../.storybook/mockups';
	import { PATHS, SIZE } from '../../logo-showcase/lib/logo';
	import { FORMATS, grid, type Ground, ICON, LOCKUP } from '../posts/post';
	import PostFrame from '../posts/PostFrame.svelte';
	import { type Scene, SCENE_MODEL_ID, SCENES } from './scenes';
	import WindowsTray from './WindowsTray.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children?: Snippet;
		headline: string;
		kind: 'art' | 'brand' | 'card' | 'desktop' | 'window';
		lede?: string;
		overlay?: boolean;
		/** the app scene a `window` shows */
		scene?: Scene;
		tone: Ground;
	}

	let { children, headline, kind, lede, overlay, scene = 'chat', tone }: Props = $props();

	const format = FORMATS.msScreenshot;
	const g = grid(format);
	// the quick chat is 28 units wide; the desktop is drawn at 1.3 times Windows' size, its taskbar 48px high; the
	// Models card close up at 2.1 times
	const windowZoom = (g.unit * 28) / 1040;
	const desk = 1.3;
	const close = 2.1;
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
			<!-- laid out at 1040 px, a small laptop's, and zoomed so the app reads at Store size; it ends a unit above the edge -->
			<div
				style:margin-bottom="{(g.unit - below) / windowZoom}px"
				style:zoom={windowZoom}
				class="window"
			>
				<div class="window-head">
					<span class="app-icon">{ICON}</span>

					<span class="pill"><i class="dot"></i>{SCENE_MODEL_ID}</span>
				</div>

				<iframe
					loading="eager"
					src="iframe.html?id={SCENES[scene]}&viewMode=story&globals=interactionPace:instant"
					tabindex="-1"
					title="Llama, {scene}"
				></iframe>

				<p class="dismiss"><kbd>Esc</kbd> to dismiss</p>
			</div>
		{:else if kind === 'desktop'}
			<!-- the flyout stands on the taskbar at the screen's right edge -->
			<div style:bottom="{48 * desk - below}px" style:right="{-g.margin}px" class="flyout">
				<div style:zoom={desk}><WindowsTray rows={3} /></div>
			</div>

			<div
				style:bottom="{-below}px"
				style:left="{-g.margin}px"
				style:width="{format.width}px"
				class="taskbar"
			>
				<div style:width="{format.width / desk}px" style:zoom={desk}>
					<WindowsTaskbar clock={['10:24 AM', '11/6/2026']} mode="dark" {trayIcon} />
				</div>
			</div>
		{:else if kind === 'card'}
			<div style:margin-bottom="{-below / close}px" style:zoom={close} class="close-up">
				<WindowsTray card rows={4} />
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
	/* larger and darker than a post's lede: the Store shows screenshots at about a third of their size */
	.text :global(.t-lede) {
		margin: 0;
		color: color-mix(in srgb, var(--p-ink) 78%, transparent);
		font-size: calc(var(--u) * 0.9);
		line-height: 1.3;
		text-wrap: balance;
	}

	/* the quick chat: Windows' dark frame round Llama Chat, its model in a pill, Esc to close it */
	.window {
		display: flex;
		flex: 1 1 0;
		flex-direction: column;
		width: 1040px;
		min-height: 0;
		margin-top: 40px;
		overflow: hidden;
		border: 1px solid rgb(255 255 255 / 0.1);
		border-radius: 8px;
		background: #202020;
		color: #fff;
		font:
			400 13px / 1 'Segoe UI Variable',
			'Segoe UI',
			var(--sans);
		box-shadow:
			0 32px 64px -24px rgb(17 34 60 / 0.5),
			0 2px 8px rgb(17 34 60 / 0.12);
	}
	.window-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: 48px;
		padding: 0 12px 0 18px;
	}
	.app-icon {
		font: 700 18px / 1 var(--sans);
	}
	.pill {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 7px 12px;
		border-radius: 999px;
		background: rgb(255 255 255 / 0.06);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #3fb950;
	}
	iframe {
		display: block;
		flex: 1;
		width: calc(100% - 32px);
		margin: 0 16px;
		border: 0;
		border-radius: 4px;
		background: #0a0a0a;
		pointer-events: none;
	}
	.dismiss {
		margin: 0;
		padding: 10px 0 12px;
		color: rgb(255 255 255 / 0.6);
		font-size: 12px;
		text-align: center;
	}
	.dismiss kbd {
		color: rgb(255 255 255 / 0.85);
		font: inherit;
	}

	.close-up {
		flex: none;
		width: 400px;
		margin-top: 20px;
		border-radius: 8px;
		box-shadow: 0 24px 48px -20px rgb(17 34 60 / 0.5);
	}

	/* the desktop: the flyout over the full taskbar, which runs edge to edge at the bottom */
	.flyout {
		position: absolute;
		right: 0;
	}
	.taskbar {
		position: absolute;
	}

	/*
		the opener: the statement big in the middle, the lede at a size that still reads at the Store's thumbnail (about a
		third of the piece), and the lockup between the two in size, set apart under them
	*/
	.kind-brand {
		justify-content: center;
		gap: calc(var(--u) * 2.2);
		/* centred on the whole piece, not only on the area above the bottom quarter */
		margin-bottom: calc(-1 * var(--safe-bottom));
	}
	.kind-brand .text {
		gap: calc(var(--u) * 1.1);
	}
	.kind-brand .t-display {
		margin: 0;
		font-size: calc(var(--u) * 3.6);
		white-space: pre-line;
	}
	.kind-brand .text :global(.t-lede) {
		max-width: calc(var(--u) * 24);
		font-size: calc(var(--u) * 1.05);
	}
	.lockup {
		font: 700 calc(var(--u) * 1.6) / 1 var(--sans);
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
