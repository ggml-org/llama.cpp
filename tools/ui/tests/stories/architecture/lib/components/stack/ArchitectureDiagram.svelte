<!-- The architecture drawing, layers bottom-up: brand, library, apps, where Chat ships; backend and docs on the right -->
<script lang="ts">
	import Figure from '../ui/Figure.svelte';
	import AppsBand from './AppsBand.svelte';
	import BrandBand from './BrandBand.svelte';
	import LibraryBand from './LibraryBand.svelte';
	import ShipsRow from './ShipsRow.svelte';
	import SideColumn from './SideColumn.svelte';
</script>

<Figure
	label="Architecture: ui/lib (@ggml/llama-ui), including shared model UI, is imported by the llama.app website (llama-pages) and by apps in ui/apps: Llama Chat and other apps, plus platform examples and third-party apps. Llama Chat ships to chat.llama.app, npx llama-chat, and the build bundled in llama-server. The macOS and Windows apps ship the llama binary, whose llama serve command is llama-server with this UI built in. Apps reach llama-server by URL or paired over WebRTC. The library is documented at ui.llama.app. Brand and design language is shared across apps, docs, web and promotional content."
	minWidth={900}
>
	<svg class="arch" viewBox="0 0 1114 800" xmlns="http://www.w3.org/2000/svg">
		<defs>
			<marker
				id="stack-ah"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah" d="M0,0 L10,5 L0,10 z" /></marker
			>

			<marker
				id="stack-ah-app"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah-app" d="M0,0 L10,5 L0,10 z" /></marker
			>

			<marker
				id="stack-ah-lib"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah-lib" d="M0,0 L10,5 L0,10 z" /></marker
			>
		</defs>

		<ShipsRow />

		<AppsBand />

		<LibraryBand />

		<SideColumn />

		<BrandBand />
	</svg>
</Figure>

<style>
	/* the bands are child components, so their shapes are styled through :global under .arch */
	.arch :global(text) {
		font-family: var(--sans);
		fill: var(--fg);
	}
	.arch :global(.t) {
		font-size: 15px;
		font-weight: 600;
	}
	.arch :global(.tl) {
		font-family: var(--display);
		font-size: 26px;
		font-weight: 700;
	}
	.arch :global(.s) {
		font-size: 12.5px;
		fill: var(--muted);
	}
	.arch :global(.m) {
		font-family: var(--mono);
		font-size: 11.5px;
	}
	.arch :global(.lbl) {
		font-family: var(--mono);
		font-size: 11px;
		letter-spacing: 0.06em;
		fill: var(--muted);
	}
	.arch :global(.b-app) {
		fill: var(--app-bg);
		stroke: var(--app);
	}
	.arch :global(.b-lib) {
		fill: var(--lib-bg);
		stroke: var(--lib);
	}
	.arch :global(.b-be) {
		fill: var(--be-bg);
		stroke: var(--be);
	}
	.arch :global(.box) {
		fill: var(--surface);
		stroke: var(--line);
	}
	.arch :global(.box-app) {
		fill: var(--surface);
		stroke: var(--app);
		stroke-width: 1.6;
	}
	.arch :global(.box-lib) {
		fill: var(--surface);
		stroke: var(--lib);
	}
	.arch :global(.dash) {
		stroke-dasharray: 5 4;
	}
	.arch :global(.band) {
		stroke-width: 1;
		fill-opacity: 1;
	}
	.arch :global(.c-app) {
		fill: var(--app);
	}
	.arch :global(.c-lib) {
		fill: var(--lib);
	}
	.arch :global(.c-be) {
		fill: var(--be);
	}
	.arch :global(.wire) {
		fill: none;
		stroke: var(--line);
		stroke-width: 1.6;
	}
	.arch :global(.wire-app) {
		fill: none;
		stroke: var(--app);
		stroke-width: 1.6;
	}
	.arch :global(.wire-lib) {
		fill: none;
		stroke: var(--lib);
		stroke-width: 1.6;
	}
	.ah {
		fill: var(--line);
	}
	.ah-app {
		fill: var(--app);
	}
	.ah-lib {
		fill: var(--lib);
	}
</style>
