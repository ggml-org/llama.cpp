<script lang="ts">
	import type { Component, Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		title: string;
		Icon: Component;
	}

	let { children, title, Icon }: Props = $props();
</script>

<div>
	<div class="mb-6 flex items-center gap-2 border-b border-border/30 pb-6">
		<Icon class="h-5 w-5" />

		<h3 class="text-lg font-semibold">{title}</h3>
	</div>

	<div class="space-y-6">
		{@render children()}
	</div>
</div>
