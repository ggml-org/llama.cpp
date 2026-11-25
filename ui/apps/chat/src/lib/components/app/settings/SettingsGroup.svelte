<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title?: string;
		children: Snippet;
	}

	let { children, title }: Props = $props();
</script>

<div>
	{#if title}
		<h3 class="mb-6 text-body font-semibold">{title}</h3>
	{/if}

	<div class="space-y-8">
		{@render children()}
	</div>
</div>
