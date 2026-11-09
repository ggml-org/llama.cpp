<!-- Architecture diagram, top row: where Llama Chat ships, and the build + publish bus feeding it -->
<svelte:options namespace="svg" />

<rect
	class="box"
	height="130"
	rx="14"
	style="fill:var(--bg);stroke:var(--rule)"
	width="808"
	x="18"
	y="20"
/>
<text class="lbl" x="34" y="40">WHERE LLAMA CHAT SHIPS</text>
<rect class="box" height="84" rx="9" width="182" x="34" y="52" />
<text class="t" x="48" y="80">chat.llama.app</text>
<text class="s" x="48" y="101">Hosted static build;</text>
<text class="s" x="48" y="118">user adds their own server</text>
<rect class="box" height="84" rx="9" width="182" x="230" y="52" />
<text class="m" style="font-weight:500" x="244" y="80">npx llama-chat</text>
<text class="s" x="244" y="101">UI only, from npm;</text>
<text class="s" x="244" y="118">you add a server</text>
<rect class="box" height="84" rx="9" width="182" x="426" y="52" />
<text class="t" x="440" y="80">llama-server</text>
<text class="s" x="440" y="101">Built-in web UI,</text>
<text class="s" x="440" y="118">same as today</text>
<rect class="box dash" height="84" rx="9" width="190" x="622" y="52" />
<text class="t" x="636" y="80">macOS + Windows apps</text>
<text class="s" x="636" y="101">llama binary: llama serve</text>
<text class="s" x="636" y="118">is llama-server + this UI</text>

<!-- publish bus -->
<path class="wire-app" d="M127,228 V176" />
<path class="wire-app" d="M125,176 H717" />
<path class="wire-app" d="M125,176 V139" marker-end="url(#stack-ah-app)" />
<path class="wire-app" d="M321,176 V139" marker-end="url(#stack-ah-app)" />
<path class="wire-app" d="M517,176 V139" marker-end="url(#stack-ah-app)" />
<path class="wire-app" d="M717,176 V139" marker-end="url(#stack-ah-app)" />
<text class="lbl" style="fill:var(--app)" x="135" y="170">LLAMA CHAT: BUILD + PUBLISH</text>
