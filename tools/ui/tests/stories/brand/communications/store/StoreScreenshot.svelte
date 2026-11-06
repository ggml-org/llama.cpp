<!--
	A Microsoft Store screenshot: one practical point in a centred headline, and under it the thing that proves it,
	large, as the Store's best listings do. Drawn after Llama for Windows as it is, in Windows' dark theme: `window` is
	its quick chat, a frameless window with the running model in a pill and Llama Chat inside (a real app scene, a
	story of Store scenes); `overlay` is that quick chat as Alt + Space opens it, over another app on the desktop, with
	the shortcut's keys under the words; `desktop` is its tray flyout over the taskbar, flush with the screen's edge;
	`card` is the flyout's Models card up close (its list, or a model's page with fit verdicts), running off the bottom;
	`brand` is the opener, the headline at display size over
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
		kind: 'art' | 'brand' | 'card' | 'desktop' | 'overlay' | 'window';
		lede?: string;
		overlay?: boolean;
		/** the app scene a `window` shows */
		scene?: Scene;
		tone: Ground;
		/** what the `card` close-up shows: the model list, or a model's page */
		tray?: 'details' | 'list';
	}

	let {
		children,
		headline,
		kind,
		lede,
		overlay,
		scene = 'chat',
		tone,
		tray = 'list'
	}: Props = $props();

	const format = FORMATS.msScreenshot;
	const g = grid(format);
	// the quick chat is 28 units wide; the desktop is drawn at 1.6 times Windows' size, its taskbar 48px high; the
	// Models card close up at 2.1 times
	const windowZoom = (g.unit * 28) / 1160;
	const desk = 1.6;
	const close = 2.1;
	// from the area's bottom down to the piece's edge
	const below = g.margin + (format.safe?.bottom ?? 0);
	// on the desktop the shot runs down to a margin above the taskbar, so the words centre on the screen over it; the
	// flyout and the taskbar are placed from there
	const deskEdge = g.margin + 48 * desk;
	const trayIcon = { paths: PATHS, viewBox: `0 0 ${SIZE} ${SIZE}` };
	// Alt + Space: the quick chat, 26 units wide, over another app's window, a gap above the taskbar (drawn at 1.4)
	const bar = 1.4;
	const floatWidth = g.unit * 26;
	const floatZoom = floatWidth / 1160;
	let floatHeight = $state(0);
</script>

{#snippet quickChat(sceneName: Scene)}
	<div class="window-head">
		<span class="app-icon">{ICON}</span>

		<span class="pill"><i class="dot"></i>{SCENE_MODEL_ID}</span>
	</div>

	<iframe
		loading="eager"
		src="iframe.html?id={SCENES[sceneName]}&viewMode=story&globals=interactionPace:instant"
		tabindex="-1"
		title="Llama, {sceneName}"
	></iframe>

	<p class="dismiss"><kbd>Esc</kbd> to dismiss</p>
{/snippet}

{#snippet taskbar(zoom: number, bottom: number)}
	<div
		style:bottom="{-bottom}px"
		style:left="{-g.margin}px"
		style:width="{format.width}px"
		class="taskbar"
	>
		<div style:width="{format.width / zoom}px" style:zoom>
			<WindowsTaskbar clock={['10:24 AM', '11/6/2026']} mode="dark" {trayIcon} />
		</div>
	</div>
{/snippet}

<PostFrame {format} label="Store screenshot, {kind}, {tone}" {overlay} {tone}>
	<!-- beside the flyout, the words stand in the middle of the screen above the taskbar -->
	<div
		style:margin-bottom={kind === 'desktop' ? `${deskEdge - below}px` : undefined}
		class={['shot', `kind-${kind}`]}
	>
		<div class="text">
			<h2 class={kind === 'brand' ? 't-display' : 'headline'}>{headline}</h2>

			{#if lede}<p class="t-lede">{lede}</p>{/if}

			{#if kind === 'overlay'}
				<span class="keys"><kbd>Alt</kbd>+<kbd>Space</kbd></span>
			{/if}
		</div>

		{#if kind === 'window'}
			<!-- laid out at 1160 px, a small laptop's, and zoomed so the app reads at Store size; it ends a unit above the edge -->
			<div
				style:margin-bottom="{(g.unit - below) / windowZoom}px"
				style:zoom={windowZoom}
				class="window"
			>
				{@render quickChat(scene)}
			</div>
		{:else if kind === 'desktop'}
			<!-- the flyout stands on the taskbar at the screen's right edge -->
			<div style:bottom="{48 * desk - deskEdge}px" style:right="{-g.margin}px" class="flyout">
				<div style:zoom={desk}><WindowsTray /></div>
			</div>

			{@render taskbar(desk, deskEdge)}
		{:else if kind === 'overlay'}
			<!-- the stage runs down to a gap above the taskbar; another app's window behind, the quick chat in front -->
			<div style:margin-bottom="{-(below - 48 * bar - 16)}px" class="stage">
				<div aria-hidden="true" class="app">
					<div class="app-bar"><i></i><i></i><i></i></div>

					<div class="app-page">
						<i class="h"></i>{#each [92, 100, 96, 74, 0, 100, 88, 97, 60] as w, i (i)}<i
								style:width="{w}%"
							></i>{/each}
					</div>
				</div>

				<div bind:clientHeight={floatHeight} style:width="{floatWidth}px" class="float">
					{#if floatHeight}
						<div
							style:height="{floatHeight / floatZoom}px"
							style:zoom={floatZoom}
							class="window floating"
						>
							{@render quickChat('quick')}
						</div>
					{/if}
				</div>
			</div>

			{@render taskbar(bar, below)}
		{:else if kind === 'card'}
			<div style:margin-bottom="{-below / close}px" style:zoom={close} class="close-up">
				<WindowsTray card view={tray} />
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
		width: 1160px;
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

	/* the desktop: the words on the left, the flyout over the full taskbar, which runs edge to edge at the bottom */
	.kind-desktop {
		align-items: flex-start;
		justify-content: center;
	}
	.kind-desktop .text {
		justify-items: start;
		max-width: calc(var(--u) * 24);
		text-align: left;
	}
	.flyout {
		position: absolute;
		right: 0;
	}

	/* Alt + Space: the keys, then the quick chat over another app */
	.keys {
		display: flex;
		align-items: center;
		gap: calc(var(--u) * 0.3);
		margin-top: calc(var(--u) * 0.3);
		color: var(--p-muted);
		font: 500 calc(var(--u) * 0.6) / 1 var(--mono);
	}
	.keys kbd {
		padding: calc(var(--u) * 0.22) calc(var(--u) * 0.45);
		border: 1px solid var(--p-line);
		border-bottom-width: 3px;
		border-radius: calc(var(--u) * 0.22);
		background: var(--p-card);
		color: var(--p-ink);
		font: inherit;
	}
	.stage {
		position: relative;
		flex: 1;
		width: 100%;
		min-height: 0;
		margin-top: calc(var(--u) * 0.8);
	}
	.app {
		position: absolute;
		inset: 0 calc(var(--u) * 3) calc(-1 * var(--u)) calc(var(--u) * 3);
		overflow: hidden;
		border: 1px solid rgb(17 17 17 / 0.12);
		border-radius: 10px 10px 0 0;
		background: #fbfbfb;
		box-shadow: 0 20px 50px -20px rgb(17 34 60 / 0.35);
	}
	.app-bar {
		display: flex;
		justify-content: flex-end;
		gap: 28px;
		height: 44px;
		padding: 0 22px;
		align-items: center;
		background: #f0f0f0;
	}
	.app-bar i {
		width: 14px;
		height: 2px;
		background: rgb(17 17 17 / 0.5);
	}
	.app-page {
		display: grid;
		gap: 18px;
		width: 60%;
		margin: 56px auto 0;
	}
	.app-page i {
		height: 12px;
		border-radius: 6px;
		background: rgb(17 17 17 / 0.1);
	}
	.app-page i.h {
		width: 40%;
		height: 22px;
		margin-bottom: 10px;
		background: rgb(17 17 17 / 0.22);
	}
	.float {
		position: absolute;
		top: calc(var(--u) * 1.4);
		bottom: 0;
		left: 50%;
		transform: translateX(-50%);
	}
	.window.floating {
		flex: none;
		margin-top: 0;
		box-shadow:
			0 40px 80px -24px rgb(0 0 0 / 0.6),
			0 2px 8px rgb(0 0 0 / 0.2);
	}
	.taskbar {
		position: absolute;
	}

	/*
		the opener: the statement big in the middle, the lede at a size that still reads at the Store's thumbnail (about a
		third of the piece), and the lockup between the two in size, set apart under them
	*/
	/*
		centred on the whole piece, and short enough (the lede on one line) to stay clear of the bottom quarter, where the
		Store may set its own text
	*/
	.kind-brand {
		justify-content: center;
		gap: calc(var(--u) * 1.3);
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
		max-width: calc(var(--u) * 34);
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
