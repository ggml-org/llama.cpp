<!--
	A link preview. llama.app's Open Graph card, on the Product key visual: the Llama lockup, the headline and the
	caption on the hero blue or the navy, with the app beside them. llama.cpp's GitHub social preview, on the
	Statement key visual: its lockup in Llama Mono on a neutral ground, the headline, the lede and a command.
-->
<script lang="ts">
	import Product from '../layouts/Product.svelte';
	import Statement from '../layouts/Statement.svelte';
	import type { Ground, LinkPreviewCopy } from './post';

	interface Props {
		copy: LinkPreviewCopy;
		kind: 'og' | 'repo';
		overlay?: boolean;
		tone: Ground;
	}

	let { copy, kind, overlay, tone }: Props = $props();
</script>

{#if kind === 'og'}
	<Product
		caption={copy.caption}
		format="og"
		header={{ brand: 'Llama' }}
		headline={copy.headline}
		{overlay}
		{tone}
	/>
{:else}
	<Statement
		command={copy.command}
		format="repo"
		header={{ brand: 'llama.cpp', url: 'github.com/ggml-org/llama.cpp' }}
		headline={copy.repoHeadline}
		lede={copy.repoLede}
		{overlay}
		{tone}
	/>
{/if}
