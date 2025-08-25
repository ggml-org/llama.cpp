<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		title: string;
		icon: any;
	}

	let { children, title, icon }: Props = $props();
</script>

<div>
	<div class="border-border/30 flex items-center gap-2 border-b pb-6 mb-6">
		{@render icon({ class: 'h-5 w-5' })}
		<h3 class="text-lg font-semibold">{title}</h3>
	</div>

	<div class="space-y-6">
		{@render children()}
	</div>
</div>
