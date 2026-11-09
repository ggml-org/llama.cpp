<!-- The rule: a one-line statement over two columns (RuleSide), what lives in the library and what in the apps -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, title }: { title: string; children: Snippet } = $props();
</script>

<div class="rule card-surface rounded-card">
	<div class="head">
		<span class="eyebrow">The rule</span>

		<p>{title}</p>
	</div>

	<div class="cols">{@render children()}</div>
</div>

<style>
	.rule {
		padding: 18px 20px;
		display: grid;
		gap: 14px;
	}
	.head {
		display: grid;
		gap: 4px;
	}
	.eyebrow {
		font-family: var(--mono);
		font-size: 12px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.head p {
		font-family: var(--display);
		font-size: 22px;
		font-weight: 600;
		margin: 0;
		text-wrap: balance;
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 20px;
	}
	@media (max-width: 700px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}
</style>
