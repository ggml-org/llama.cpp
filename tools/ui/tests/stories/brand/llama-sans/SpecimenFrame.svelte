<!--
	The page a Llama Sans story sits on, set in one version (variant) and weight. A copy of the specimen's page template
	in ggml-org/llama.brand (fonts/specimen/src/lib/templates/Doc.svelte); the specimen's components and data in lib/
	are copied from there too, so change them there first.
-->
<script lang="ts">
	import '../fonts/fonts.css';
	import './lib/styles/specimen.css';
	import { fallback } from './lib/specimen';
	import type { Snippet } from 'svelte';

	let {
		children,
		variant,
		weight = 400
	}: { children: Snippet; variant: string; weight?: number } = $props();
</script>

<div
	style:--display={`"Llama Sans ${variant}", ${fallback}`}
	style:--w={weight}
	class="specimen wrap"
>
	{@render children()}
</div>

<style>
	.wrap {
		display: grid;
		gap: 56px;
		max-width: 1180px;
		margin: 0 auto;
		padding: 40px max(16px, env(safe-area-inset-right, 0px)) 80px
			max(16px, env(safe-area-inset-left, 0px));
	}
	.wrap > :global(*) {
		margin: 0;
		min-width: 0;
	}
	/* the tester's hint points at the specimen's control bar, which the stories don't have */
	.wrap :global(.tester .hint :is(.wide, .narrow)) {
		display: none;
	}
	/* room at the bottom on phones */
	@media (max-width: 600px) {
		.wrap {
			padding-bottom: calc(80px + env(safe-area-inset-bottom, 0px));
		}
	}
</style>
