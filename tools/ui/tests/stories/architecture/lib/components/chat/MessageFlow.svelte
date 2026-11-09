<!-- How a message travels: compose, the agentic loop, chat completions, and the three places tools run -->
<script lang="ts">
	import Figure from '../ui/Figure.svelte';
</script>

<Figure
	label="In the browser, the composer sends to the agentic loop, which streams from the active llama-server's chat completions endpoint. Tool calls go to browser tools in a sandbox, to server tools on llama-server in the working directory, or to MCP servers. Results return to the loop; the reply is rendered and saved to IndexedDB."
	minWidth={880}
>
	<svg viewBox="0 0 1080 350" xmlns="http://www.w3.org/2000/svg">
		<defs
			><marker
				id="chat-arrow"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah" d="M0,0 L10,5 L0,10 z" /></marker
			></defs
		>

		<rect class="band-phone" height="306" rx="14" width="524" x="8" y="32" />

		<text class="lbl" style="fill:var(--app)" x="22" y="52">IN THE BROWSER</text>

		<rect class="band-srv" height="306" rx="14" width="254" x="546" y="32" />

		<text class="lbl" style="fill:var(--be)" x="560" y="52">ACTIVE LLAMA-SERVER</text>

		<rect class="band-ext" height="142" rx="14" width="258" x="814" y="196" />

		<text class="lbl" style="fill:var(--muted)" x="828" y="216">EXTERNAL</text>

		<rect class="node" height="130" rx="10" width="160" x="20" y="66" />

		<text class="t" x="34" y="92">Compose</text>

		<text class="s" x="34" y="116">Live-preview Markdown</text>

		<text class="s" x="34" y="133">editor, attachments,</text>

		<text class="s" x="34" y="150">@ MCP resources,</text>

		<text class="s" x="34" y="167">project + cwd chips</text>

		<rect
			class="node"
			height="130"
			rx="10"
			style="stroke:var(--app);stroke-width:1.6"
			width="324"
			x="196"
			y="66"
		/>

		<text class="t" x="210" y="92">Agentic loop</text>

		<text class="m" x="320" y="92">chatStore + agenticStore</text>

		<text class="s" x="210" y="116">Streams the reply, runs each tool call the model</text>

		<text class="s" x="210" y="133">makes, feeds the result back, repeats.</text>

		<text class="s" x="210" y="160">Permission gates: once, always, deny.</text>

		<text class="m" x="210" y="180">model, server and tools from the chat</text>

		<rect class="node" height="130" rx="10" width="230" x="558" y="66" />

		<text class="t" x="572" y="92">Chat completions</text>

		<text class="m" x="572" y="112">/v1/chat/completions</text>

		<text class="s" x="572" y="138">Streamed. Local, by URL,</text>

		<text class="s" x="572" y="155">or a --connect tunnel.</text>

		<path class="wire" d="M180,131 H194" marker-end="url(#chat-arrow)" />

		<path class="wire" d="M520,112 H556" marker-end="url(#chat-arrow)" />

		<path class="wire" d="M558,150 H522" marker-end="url(#chat-arrow)" />

		<rect class="node" height="100" rx="10" width="160" x="20" y="226" />

		<text class="t" x="34" y="252">Render + save</text>

		<text class="s" x="34" y="274">Markdown, reasoning,</text>

		<text class="s" x="34" y="291">KaTeX, Mermaid, code</text>

		<text class="m" x="34" y="313">IndexedDB (Dexie)</text>

		<path class="wire" d="M230,196 V276 H182" marker-end="url(#chat-arrow)" />

		<rect class="node" height="100" rx="10" width="270" x="250" y="226" />

		<text class="t" x="264" y="252">Browser tools</text>

		<text class="s" x="264" y="274">run_javascript in a sandbox,</text>

		<text class="s" x="264" y="291">read_media, get_datetime,</text>

		<text class="s" x="264" y="308">project tools</text>

		<path class="wire" d="M385,196 V224" marker-end="url(#chat-arrow)" />

		<rect class="node" height="100" rx="10" width="230" x="558" y="226" />

		<text class="t" x="572" y="252">Server tools</text>

		<text class="s" x="572" y="274">read, write, edit file; grep,</text>

		<text class="s" x="572" y="291">glob, exec_shell_command</text>

		<text class="m" x="572" y="308">in the working directory</text>

		<rect class="node" height="100" rx="10" width="234" x="826" y="226" />

		<text class="t" x="840" y="252">MCP servers</text>

		<text class="s" x="840" y="274">Their tools, plus resources</text>

		<text class="s" x="840" y="291">you @-mention in a message</text>

		<path class="wire" d="M500,196 V212 H540 V276 H556" marker-end="url(#chat-arrow)" />

		<path class="wire" d="M540,212 H812 V276 H824" marker-end="url(#chat-arrow)" />

		<text class="m" x="600" y="208">tool calls</text>
	</svg>
</Figure>

<style>
	text {
		font-family: var(--sans);
		fill: var(--fg);
	}
	.t {
		font-size: 16px;
		font-weight: 600;
	}
	.s {
		font-size: 12.5px;
		fill: var(--fg);
	}
	.m {
		font-family: var(--mono);
		font-size: 11px;
		fill: var(--muted);
	}
	.lbl {
		font-family: var(--mono);
		font-size: 11px;
		letter-spacing: 0.06em;
	}
	.band-phone {
		fill: var(--app-bg);
		stroke: var(--app);
	}
	.band-srv {
		fill: var(--be-bg);
		stroke: var(--be);
	}
	.band-ext {
		fill: var(--bg);
		stroke: var(--line);
		stroke-dasharray: 5 4;
	}
	.node {
		fill: var(--surface);
		stroke: var(--line);
	}
	.wire {
		fill: none;
		stroke: var(--fg);
		stroke-width: 1.6;
	}
	.ah {
		fill: var(--fg);
	}
</style>
