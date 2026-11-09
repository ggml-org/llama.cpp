<!-- One build per release, fanned out to the three channels -->
<script lang="ts">
	import Figure from '../ui/Figure.svelte';
</script>

<Figure
	label="One CI job per llama.cpp release builds dist with adapter-static. The same dist goes to three channels: uploaded to the Hugging Face bucket that llama-server's CMake build embeds; published to npm as llama-chat, the UI alone served on localhost; and deployed to chat.llama.app on static hosting. Only the first comes with a llama-server; the other two use the server you add."
>
	<svg viewBox="0 0 860 330" xmlns="http://www.w3.org/2000/svg">
		<defs
			><marker
				id="release-arrow"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah" d="M0,0 L10,5 L0,10 z" /></marker
			></defs
		>

		<text class="lbl" x="20" y="28">ONCE PER LLAMA.CPP RELEASE</text>

		<rect class="node-lib" height="96" rx="10" width="230" x="20" y="120" />

		<text class="t" x="36" y="148">Build</text>

		<text class="m" x="36" y="170">vite build · adapter-static</text>

		<text class="s" x="36" y="196">one dist/, hash router,</text>

		<text class="s" x="36" y="212">relative paths</text>

		<path class="wire" d="M250,168 H300" />

		<path class="wire" d="M300,74 V262" />

		<path class="wire" d="M300,74 H338" marker-end="url(#release-arrow)" />

		<path class="wire" d="M300,168 H338" marker-end="url(#release-arrow)" />

		<path class="wire" d="M300,262 H338" marker-end="url(#release-arrow)" />

		<text class="m" x="260" y="160">same dist</text>

		<text class="lbl" x="340" y="28">CHANNEL</text>

		<rect class="node-app" height="70" rx="10" width="500" x="340" y="40" />

		<text class="t" x="356" y="66">Bundled in llama-server</text>

		<text class="s" x="356" y="88"
			>via the HF bucket; native apps: llama serve in the llama binary</text
		>

		<rect class="node-app" height="70" rx="10" width="500" x="340" y="134" />

		<text class="t" x="356" y="160">npx llama-chat</text>

		<text class="s" x="356" y="182"
			>npm package llama-chat: the UI only, served from localhost; add your server</text
		>

		<rect class="node-app" height="70" rx="10" width="500" x="340" y="228" />

		<text class="t" x="356" y="254">chat.llama.app</text>

		<text class="s" x="356" y="276"
			>the UI only, nothing to install; it uses the server you add</text
		>
	</svg>
</Figure>

<style>
	text {
		font-family: var(--sans);
		fill: var(--fg);
	}
	.t {
		font-size: 15px;
		font-weight: 600;
	}
	.s {
		font-size: 12.5px;
		fill: var(--muted);
	}
	.m {
		font-family: var(--mono);
		font-size: 11.5px;
		fill: var(--muted);
	}
	.lbl {
		font-family: var(--mono);
		font-size: 11px;
		letter-spacing: 0.06em;
		fill: var(--muted);
	}
	.node-lib {
		fill: var(--lib-bg);
		stroke: var(--lib);
		stroke-width: 1.6;
	}
	.node-app {
		fill: var(--surface);
		stroke: var(--app);
		stroke-width: 1.4;
	}
	.wire {
		fill: none;
		stroke: var(--line);
		stroke-width: 1.6;
	}
	.ah {
		fill: var(--line);
	}
</style>
