<!--
	One layer of the code: its name, a tag for where it lands, then Markdown.
	`kind`: new (stays in the app), reuse (goes to the library), mix (split), ext (outside).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let {
		children,
		kind,
		name,
		tag
	}: { name: string; tag: string; kind: 'new' | 'reuse' | 'mix' | 'ext'; children: Snippet } =
		$props();
</script>

<div class="ly card-surface rounded-card">
	<div class="lh"><b>{name}</b><span class={['tag', kind]}>{tag}</span></div>

	<div class="body">{@render children()}</div>
</div>

<style>
	.ly {
		padding: 12px 14px;
		display: grid;
		grid-template-columns: 200px 1fr;
		gap: 12px;
		align-items: start;
	}
	@media (max-width: 700px) {
		.ly {
			grid-template-columns: 1fr;
		}
	}
	.lh {
		display: grid;
		gap: 6px;
		justify-items: start;
	}
	.tag {
		font-family: var(--mono);
		font-size: 10.5px;
		padding: 2px 7px;
		border-radius: 4px;
		white-space: nowrap;
	}
	.new {
		background: var(--app-bg);
		color: var(--app);
	}
	.reuse {
		background: var(--lib-bg);
		color: var(--lib);
	}
	.mix {
		background: var(--be-bg);
		color: var(--be);
	}
	.ext {
		background: var(--bg);
		color: var(--muted);
		border: 1px solid var(--rule);
	}
	.body {
		font-size: 13.5px;
		color: var(--muted);
	}
	.body :global(*) {
		margin: 0;
	}
	.body :global(code) {
		color: var(--fg);
	}
</style>
