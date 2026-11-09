<!-- Architecture diagram, library band: Llama UI (@ggml/llama-ui) and the import arrows up to the apps -->
<svelte:options namespace="svg" />

<!-- import arrows -->
<path class="wire-lib" d="M127,490 V455" marker-end="url(#stack-ah-lib)" />
<path class="wire-lib" d="M421,490 V455" marker-end="url(#stack-ah-lib)" />
<path class="wire-lib" d="M718,490 V455" marker-end="url(#stack-ah-lib)" />
<text class="lbl" style="fill:var(--lib)" x="135" y="477">IMPORT</text>

<rect class="b-lib band" height="176" rx="14" width="808" x="18" y="492" />
<text class="lbl" style="fill:var(--lib)" x="34" y="512"
	>LIBRARY · llama.cpp/ui/lib · NO BUSINESS LOGIC</text
>
<text class="tl" x="34" y="546">Llama UI</text>
<text class="m" style="fill:var(--lib)" x="156" y="546">@ggml/llama-ui</text>
<text class="s" x="300" y="546">UI and UX only: components, tokens, interaction logic.</text>
<rect class="box-lib" height="88" rx="9" width="148" x="34" y="564" />
<text class="t" x="46" y="590">Design tokens</text>
<text class="s" x="46" y="611">Colour, type, spacing,</text>
<text class="s" x="46" y="628">themes, brand assets</text>
<rect class="box-lib" height="88" rx="9" width="148" x="190" y="564" />
<text class="t" x="202" y="590">Components</text>
<text class="s" x="202" y="611">Buttons, forms, dialogs</text>
<text class="s" x="202" y="628">Svelte + shadcn-svelte</text>
<rect class="box-lib" height="88" rx="9" style="stroke-width:2" width="148" x="346" y="564" />
<text class="t" x="358" y="590">Model UI</text>
<text class="s" x="358" y="611">Catalog, model card,</text>
<text class="s" x="358" y="628">quant picker, fit check</text>
<rect class="box-lib" height="88" rx="9" width="148" x="502" y="564" />
<text class="t" x="514" y="590">UX patterns</text>
<text class="s" x="514" y="611">Thread, composer,</text>
<text class="s" x="514" y="628">attachments</text>
<rect class="box-lib" height="88" rx="9" width="148" x="658" y="564" />
<text class="t" x="670" y="590">UX logic</text>
<text class="s" x="670" y="611">Flows, interactions,</text>
<text class="s" x="670" y="628">small computations</text>
