<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		children: Snippet;
	}

	let { title, children }: Props = $props();
</script>

<div>
	<h3 class="mb-6 text-base font-semibold">{title}</h3>

	<div class="space-y-8">
		{@render children()}
	</div>
</div>
