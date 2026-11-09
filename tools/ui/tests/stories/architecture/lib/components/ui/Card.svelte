<!--
	One card, on Llama UI's card surface: an optional kicker (small caps label), a title, then Markdown. `planned` is for
	what isn't built yet: a dashed edge, dimmed, with a badge saying so.
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import type { Snippet } from 'svelte';

	let {
		children,
		kicker,
		planned = false,
		title
	}: { kicker?: string; planned?: boolean; title?: string; children?: Snippet } = $props();
</script>

<div class={['card rounded-card', planned ? 'planned' : 'card-surface']}>
	{#if kicker || planned}
		<p class="head">
			{#if kicker}<span class="kicker">{kicker}</span>{/if}

			{#if planned}<Badge variant="outline">Not built yet</Badge>{/if}
		</p>
	{/if}

	{#if title}<h3>{title}</h3>{/if}

	{#if children}<div class="body">{@render children()}</div>{/if}
</div>

<style>
	/* one column as wide as the card, so long code inside scrolls instead of widening the page */
	.card {
		padding: 18px 20px;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		align-content: start;
		min-width: 0;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		margin: 0;
	}
	/* not built yet: no surface of its own, a dashed edge and the content dimmed */
	.planned {
		border: 1.5px dashed color-mix(in oklab, var(--fg-muted) 45%, transparent);
	}
	.planned :is(h3, .body) {
		opacity: 0.7;
	}
	.kicker {
		font: 500 10.5px/1.3 var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--muted);
	}
	h3 {
		margin: 0;
		font-size: 15px;
	}
	/* normal flow, so text with links or code stays one paragraph; blocks get space between */
	.body {
		font-size: 14px;
		color: var(--muted);
	}
	.body > :global(*) {
		margin: 0;
	}
	.body > :global(* + *) {
		margin-top: 8px;
	}
</style>
