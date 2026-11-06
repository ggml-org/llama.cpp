<!--
	The Llama app as it shows in a link preview: the menu bar with its icon and the macOS symbols beside it (sound,
	Wi-Fi, battery, Spotlight, Control Center, drawn after the system's own), and the panel under it with the server,
	the installed models and a recommendation. Drawn at the size it takes in a 1200 × 630 card; the models carry
	their initials rather than their makers' logos.
-->
<script lang="ts">
	import { ICON } from './post';
	import { ArrowDown, ChevronRight, Copy, LayoutGrid } from '@lucide/svelte';

	const installed = [
		{ name: 'Qwen3.8', params: '27B', size: '19.0 GB' },
		{ name: 'gpt-oss', params: '20B', size: '12.1 GB' },
		{ name: 'gemma-4', params: 'E4B', size: '4.59 GB' }
	];
</script>

<div class="app">
	<div class="bar">
		<span class="menu-icon">{ICON}</span>

		<!-- the menu bar's own symbols, drawn after macOS: sound, Wi-Fi, battery, Spotlight, Control Center -->
		<svg aria-hidden="true" class="glyph" height="16" viewBox="0 0 20 16" width="20">
			<path class="filled" d="M2 5.5h3l4-3.5v12l-4-3.5H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1z" />

			<path class="stroked" d="M12 5.2a4 4 0 0 1 0 5.6M14.6 3a7.2 7.2 0 0 1 0 10" />
		</svg>

		<svg aria-hidden="true" class="glyph" height="16" viewBox="0 0 20 16" width="20">
			<path
				class="stroked"
				d="M2.2 6.3a11 11 0 0 1 15.6 0M5 9.1a7 7 0 0 1 10 0M7.8 11.9a3 3 0 0 1 4.4 0"
			/>

			<circle class="filled" cx="10" cy="14.2" r="1.3" />
		</svg>

		<svg aria-hidden="true" class="glyph" height="16" viewBox="0 0 29 16" width="29">
			<rect class="outline" height="11.5" rx="3.6" width="24" x="1" y="2.25" />

			<rect class="filled" height="7.5" rx="1.8" width="15" x="3" y="4.25" />

			<path class="nub" d="M26.5 6.2c1 .3 1.6 1 1.6 1.8s-.6 1.5-1.6 1.8z" />
		</svg>

		<svg aria-hidden="true" class="glyph" height="16" viewBox="0 0 16 16" width="16">
			<circle class="stroked" cx="6.6" cy="6.6" r="4.6" />

			<path class="stroked" d="M10.2 10.2 14.2 14.2" />
		</svg>

		<svg aria-hidden="true" class="glyph" height="16" viewBox="0 0 18 16" width="18">
			<rect class="outline" height="5.6" rx="2.8" width="14" x="2" y="1.4" />

			<circle class="filled" cx="12.8" cy="4.2" r="1.7" />

			<rect class="filled" height="5.6" rx="2.8" width="14" x="2" y="9" />

			<circle class="knob" cx="5.2" cy="11.8" r="1.7" />
		</svg>

		<span class="clock"><span>Fri 6 Nov</span><span>10:24</span></span>
	</div>

	<div class="panel">
		<div class="head">
			<p class="title">Llama <i class="live"></i></p>

			<p class="host">localhost:8080 <Copy size={12} /></p>

			<span class="open">Open chat</span>
		</div>

		<p class="section">Installed models</p>

		{#each installed as m (m.name)}
			<div class="model">
				<span class="avatar">{m.name[0]}</span>

				<span class="info"><b>{m.name} <em>{m.params}</em></b><small>{m.size}</small></span>

				<ChevronRight size={14} />
			</div>
		{/each}

		<p class="section">Recommended for this Mac</p>

		<div class="model">
			<span class="avatar">g</span>

			<span class="info"><b>gemma-4 <em>12B</em></b><small>7.22 GB</small></span>

			<ArrowDown size={14} />
		</div>

		<div class="model browse">
			<span class="avatar"><LayoutGrid size={13} /></span>

			<span class="info"><b>Browse models</b><small>Full catalog on the web</small></span>
		</div>

		<div class="foot">
			<span class="build">llama.cpp b6123</span>

			<span class="button">Settings</span>

			<span class="button">Quit</span>
		</div>
	</div>
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 10px;
		width: 380px;
		color: #111111;
		font: 400 13px / 1.3 var(--sans);
	}
	.app :global(p) {
		margin: 0;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 15px;
		color: var(--p-ink);
		font-size: 13.5px;
		font-weight: 500;
		letter-spacing: 0.01em;
	}
	/* the symbols: bold strokes and solid shapes in the bar's ink, at the menu bar's 16px */
	.glyph {
		flex: none;
		overflow: visible;
	}
	.glyph .filled,
	.glyph .nub {
		fill: currentColor;
	}
	.glyph .nub {
		opacity: 0.45;
	}
	.glyph .stroked,
	.glyph .outline {
		fill: none;
		stroke: currentColor;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.glyph .stroked {
		stroke-width: 2;
	}
	.glyph .outline {
		stroke-width: 1.2;
		opacity: 0.5;
	}
	.glyph .knob {
		fill: var(--p-base, #fff);
	}
	.clock {
		display: flex;
		gap: 9px;
		white-space: nowrap;
	}
	.menu-icon {
		display: grid;
		place-items: center;
		width: 30px;
		height: 22px;
		border-radius: 6px;
		background: var(--p-inset);
		font: 700 15px / 1 var(--sans);
	}
	.panel {
		display: grid;
		width: 100%;
		box-sizing: border-box;
		padding: 14px;
		border: 1px solid rgb(17 17 17 / 0.08);
		border-radius: 16px;
		background: rgb(255 255 255 / 0.96);
		box-shadow:
			0 24px 60px -20px rgb(17 34 60 / 0.35),
			0 2px 6px rgb(17 34 60 / 0.08);
	}
	.head {
		display: grid;
		grid-template-columns: 1fr auto;
		row-gap: 2px;
		padding-bottom: 10px;
		border-bottom: 1px solid rgb(17 17 17 / 0.08);
	}
	.title {
		display: flex;
		align-items: center;
		gap: 6px;
		font-weight: 600;
		font-size: 15px;
	}
	.live {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--art-green);
	}
	.host {
		display: flex;
		align-items: center;
		gap: 4px;
		grid-row: 2;
		color: rgb(17 17 17 / 0.6);
	}
	.open {
		grid-row: 2;
		color: var(--art-blue);
		font-weight: 500;
	}
	.section {
		padding: 10px 0 4px;
		color: rgb(17 17 17 / 0.55);
		font-size: 12px;
	}
	.model {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 5px 0;
		color: rgb(17 17 17 / 0.4);
	}
	.avatar {
		display: grid;
		flex: none;
		place-items: center;
		width: 26px;
		height: 26px;
		border-radius: 50%;
		background: var(--neutral-150);
		color: #111111;
		font: 600 12px / 1 var(--mono);
	}
	.info {
		display: grid;
		flex: 1;
		color: #111111;
	}
	.info b {
		font-weight: 500;
	}
	.info em {
		margin-left: 4px;
		padding: 1px 5px;
		border: 1px solid rgb(17 17 17 / 0.12);
		border-radius: 4px;
		color: rgb(17 17 17 / 0.6);
		font: 400 11px var(--mono);
	}
	.info small {
		color: rgb(17 17 17 / 0.55);
		font-size: 12px;
	}
	.browse {
		margin-top: 6px;
		padding-top: 10px;
		border-top: 1px solid rgb(17 17 17 / 0.08);
	}
	.foot {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 8px;
		padding-top: 10px;
		border-top: 1px solid rgb(17 17 17 / 0.08);
	}
	.build {
		flex: 1;
		color: rgb(17 17 17 / 0.55);
		font: 400 12px var(--mono);
	}
	.button {
		padding: 3px 9px;
		border: 1px solid rgb(17 17 17 / 0.14);
		border-radius: 6px;
		font-size: 12px;
	}
</style>
