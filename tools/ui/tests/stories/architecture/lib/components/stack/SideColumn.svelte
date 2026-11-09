<!-- Architecture diagram, right column: the key, the inference backend and the docs site -->
<svelte:options namespace="svg" />

<text class="lbl" x="860" y="40">KEY</text>
<rect class="c-lib" height="14" rx="3" width="14" x="860" y="54" /><text class="s" x="884" y="66"
	>Library (one package)</text
>
<rect class="c-app" height="14" rx="3" width="14" x="860" y="80" /><text class="s" x="884" y="92"
	>Apps built on the library</text
>
<rect class="c-be" height="14" rx="3" width="14" x="860" y="106" /><text class="s" x="884" y="118"
	>Backends and docs</text
>
<rect class="box dash" height="10" rx="2" style="fill:none" width="14" x="860" y="128" /><text
	class="s"
	x="884"
	y="138">Lives outside the llama.cpp repo</text
>

<rect class="b-be band" height="256" rx="14" width="236" x="860" y="196" />
<text class="lbl" style="fill:var(--be)" x="876" y="216">INFERENCE BACKEND</text>
<text class="s" x="876" y="240">Every app talks to it the same way</text>
<rect class="box" height="70" rx="8" width="204" x="876" y="256" />
<text class="t" x="890" y="284">llama-server</text>
<text class="s" x="890" y="306">by URL: https:// or localhost</text>
<rect class="box" height="70" rx="8" width="204" x="876" y="340" />
<text class="t" x="890" y="368">llama-server --connect</text>
<text class="s" x="890" y="390">paired by code/QR, WebRTC</text>
<path class="wire" d="M828,324 H857" marker-end="url(#stack-ah)" />

<rect class="b-be band" height="176" rx="14" width="236" x="860" y="492" />
<text class="lbl" style="fill:var(--be)" x="876" y="512">DOCS</text>
<text class="t" style="font-size:17px" x="876" y="544">ui.llama.app</text>
<text class="s" x="876" y="568">Storybook for the library:</text>
<text class="s" x="876" y="586">every component and state,</text>
<text class="s" x="876" y="604">usage docs, visual tests.</text>
<text class="s" x="876" y="628">Agents can prototype UI</text>
<text class="s" x="876" y="646">variants and screenshot them.</text>
<path class="wire" d="M828,580 H857" marker-end="url(#stack-ah)" />
