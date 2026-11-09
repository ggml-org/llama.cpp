<!-- Architecture diagram, apps band: Llama Chat and other apps in ui/apps, the llama.app website and third-party apps -->
<svelte:options namespace="svg" />

<rect class="b-app band" height="256" rx="14" width="808" x="18" y="196" />
<text class="lbl" style="fill:var(--app)" x="34" y="216"
	>APPS · llama.cpp/ui/apps · BUSINESS LOGIC LIVES HERE</text
>

<a href="/chat/">
	<rect class="box-app" height="100" rx="9" width="187" x="34" y="228" />

	<text class="tl" style="font-size:19px" x="48" y="256">Llama Chat</text>

	<text class="m" style="fill:var(--app);text-decoration:underline" x="48" y="276"
		>ui/apps/chat ↗</text
	>

	<text class="s" x="48" y="299">Tools, MCP, projects;</text>

	<text class="s" x="48" y="316">manages models, servers.</text>
</a>

<rect class="box-app" height="100" rx="9" width="581" x="231" y="228" />
<text class="tl" style="font-size:19px" x="245" y="256">Other apps</text>
<text class="m" style="fill:var(--app)" x="245" y="276">ui/apps/&lt;app&gt;</text>
<text class="s" x="245" y="299"
	>More first-party apps on the same parts, once Chat is in place.</text
>
<text class="s" x="245" y="316">Each keeps its own business logic and talks to llama-server.</text>

<rect class="box-app" height="100" rx="9" style="stroke-width:1" width="187" x="34" y="340" />
<text class="t" x="48" y="368">Platform examples</text>
<text class="m" x="48" y="388">ui/apps/&lt;example&gt;</text>
<text class="s" x="48" y="411">Many, any use case.</text>
<text class="s" x="48" y="428">For developers, unpolished.</text>

<rect
	class="box-app dash"
	height="100"
	rx="9"
	style="stroke-width:1.2"
	width="187"
	x="231"
	y="340"
/>
<text class="t" x="245" y="368">llama.app website</text>
<text class="m" x="245" y="388">ggml-org/llama-pages</text>
<text class="s" x="245" y="411">Model catalog and model</text>
<text class="s" x="245" y="428">pages from the same parts.</text>

<rect
	class="box-app dash"
	height="100"
	rx="9"
	style="stroke-width:1.2"
	width="384"
	x="428"
	y="340"
/>
<text class="t" x="442" y="368">Your app</text>
<text class="m" x="442" y="388">npm i @ggml/llama-ui</text>
<text class="s" x="442" y="411">Outside the repo: third-party tools and partner products</text>
<text class="s" x="442" y="428">built on the same library and design language.</text>
