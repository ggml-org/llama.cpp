<!-- One column of the Rule: a coloured label (library or apps) and Markdown text -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, side, title }: { side: 'lib' | 'app'; title: string; children: Snippet } =
		$props();
</script>

<div class="side">
	<b class={side}>{title}</b>

	<div class="body">{@render children()}</div>
</div>

<style>
	.lib {
		color: var(--lib);
	}
	.app {
		color: var(--app);
	}
	.body {
		margin-top: 4px;
		color: var(--muted);
		font-size: 14px;
	}
	.body :global(*) {
		margin: 0;
	}
</style>
