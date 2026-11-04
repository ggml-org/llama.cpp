<!--
	A labelled section, after rsms.me/inter: a short label in the left column, the content to its right (stacked on
	phones), in one column or as many as fit (columns). The content is Markdown or components.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		columns = false,
		label
	}: { children: Snippet; columns?: boolean; label: string } = $props();
</script>

<section class="section">
	<h2>{label}</h2>

	<div class:columns class="content">{@render children()}</div>
</section>

<style>
	.section {
		display: grid;
		grid-template-columns: minmax(160px, 1fr) 3fr;
		gap: 16px 40px;
		width: var(--column);
		margin-inline: auto;
	}
	h2 {
		margin: 0;
		font: 600 16px / 1.5 var(--ui);
	}
	.content {
		display: grid;
		gap: 16px;
		align-content: start;
		min-width: 0;
		font: 400 16px / 1.5 var(--ui);
	}
	.content.columns {
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
		gap: 16px 40px;
	}
	.content > :global(*) {
		margin: 0;
	}
	.content > :global(p) {
		max-width: 68ch;
	}
	@media (max-width: 700px) {
		.section {
			grid-template-columns: 1fr;
		}
	}
</style>
