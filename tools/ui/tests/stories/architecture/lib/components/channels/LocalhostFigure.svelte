<!--
	Why npx llama-chat matters next to chat.llama.app: a llama-server on the LAN answers on plain http. chat.llama.app,
	an https page, can't call it (the browser blocks mixed content); the same UI served by npx from http://localhost can.
-->
<script lang="ts">
	import Figure from '../ui/Figure.svelte';
</script>

{#snippet browser(x: number, y: number, url: string)}
	<rect class="body" height="112" rx="10" width="290" {x} {y} />

	<path class="detail" d="M{x},{y + 30} H{x + 290}" />

	{#each [0, 1, 2] as i (i)}<circle class="dot" cx={x + 16 + i * 12} cy={y + 15} r="3.5" />{/each}

	<rect class="bar" height="18" rx="9" width="200" x={x + 62} y={y + 6} />

	<text class="url" x={x + 74} y={y + 19}>{url}</text>

	<rect class="pane" height="81" width="70" x={x + 1} y={y + 30} />

	<path
		class="text-line"
		d="M{x + 12},{y + 48} H{x + 58} M{x + 12},{y + 64} H{x + 50} M{x + 12},{y + 80} H{x + 54}"
	/>

	<rect class="bubble" height="18" rx="6" width="80" x={x + 196} y={y + 40} />

	<path class="text-line" d="M{x + 86},{y + 74} H{x + 230} M{x + 86},{y + 90} H{x + 190}" />
{/snippet}

<Figure
	label="A llama-server on the home network at http://192.168.1.20:8080, plain http. chat.llama.app is an https page, so the browser blocks its calls to that server as mixed content. npx llama-chat serves the same UI from http://localhost:3000, an http page, which can call the server."
	minWidth={720}
>
	<svg viewBox="0 0 900 420" xmlns="http://www.w3.org/2000/svg">
		<defs>
			<marker
				id="localhost-arrow"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah" d="M0,0 L10,5 L0,10 z" /></marker
			>
		</defs>

		<!-- the hosted site: an https page, its call to the LAN blocked -->
		<text class="lbl" x="20" y="22">CHAT.LLAMA.APP</text>

		{@render browser(20, 34, 'https://chat.llama.app')}

		<path class="call blocked" d="M318,90 C420,90 500,116 590,146" />

		<g class="cross" transform="translate(486,118)">
			<circle r="15" />

			<path d="M-6,-6 L6,6 M6,-6 L-6,6" />
		</g>

		<text class="s" x="512" y="104">blocked: an https page can't</text>

		<text class="s" x="512" y="120">call a plain http server</text>

		<!-- npx: the terminal that starts it, and the same UI on http://localhost -->
		<text class="lbl" x="20" y="182">NPX LLAMA-CHAT</text>

		<rect class="term" height="78" rx="10" width="290" x="20" y="194" />

		<text class="cmd" x="34" y="218"><tspan class="prompt">$</tspan> npx llama-chat --server</text>

		<text class="cmd" x="58" y="236">http://192.168.1.20:8080</text>

		<text class="cmd out" x="34" y="258">Llama Chat on http://localhost:3000</text>

		<path class="wire" d="M165,272 V290" marker-end="url(#localhost-arrow)" />

		{@render browser(20, 296, 'http://localhost:3000')}

		<path class="call" d="M318,352 C450,352 560,286 676,262" marker-end="url(#localhost-arrow)" />

		<g class="tick" transform="translate(486,320)">
			<circle r="15" />

			<path d="M-6,0 L-2,5 L7,-5" />
		</g>

		<text class="s" x="512" y="344">allowed: http to http,</text>

		<text class="s" x="512" y="360">from localhost</text>

		<!-- the server on the LAN: a small workstation, front on -->
		<rect class="body" height="150" rx="12" width="150" x="690" y="140" />

		{#each [0, 1, 2, 3, 4] as i (i)}
			<path class="vent" d="M712,{166 + i * 12} H818" />
		{/each}

		<circle class="led" cx="818" cy="270" r="4" />

		<text class="t" x="690" y="318">llama-server</text>

		<text class="m" x="690" y="338">192.168.1.20:8080</text>

		<text class="s" x="690" y="358">on the home network,</text>

		<text class="s" x="690" y="374">plain http</text>
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
	.m,
	.url {
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

	.body {
		fill: var(--surface);
		stroke: var(--fg);
		stroke-width: 1.6;
	}
	.detail,
	.vent {
		fill: none;
		stroke: var(--fg);
		stroke-width: 1.4;
		opacity: 0.35;
	}
	.vent {
		stroke-width: 3;
		stroke-linecap: round;
		opacity: 0.2;
	}
	.dot {
		fill: var(--line);
	}
	.bar,
	.pane,
	.bubble {
		fill: var(--be-bg);
	}
	.text-line {
		fill: none;
		stroke: var(--fg);
		stroke-width: 5;
		stroke-linecap: round;
		opacity: 0.18;
	}
	.led {
		fill: var(--art-cinder);
	}

	/* the terminal, in the page's code colours */
	.term {
		fill: var(--surface-code, var(--be-bg));
		stroke: var(--line);
	}
	.cmd {
		font-family: var(--mono);
		font-size: 12px;
	}
	.cmd.out,
	.prompt {
		fill: var(--muted);
	}

	/* the calls: solid where they get through, dashed and stopped by a cinder cross where they don't */
	.call,
	.wire {
		fill: none;
		stroke: var(--fg);
		stroke-width: 1.8;
	}
	.wire {
		stroke: var(--line);
	}
	.call.blocked {
		stroke: var(--muted);
		stroke-dasharray: 6 6;
		opacity: 0.6;
	}
	.ah {
		fill: var(--fg);
	}
	.cross circle {
		fill: var(--art-cinder);
	}
	.tick circle {
		fill: var(--fg);
	}
	.cross path,
	.tick path {
		fill: none;
		stroke: var(--bg);
		stroke-width: 2.4;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
</style>
