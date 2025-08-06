<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		title: string;
		icon: any;
	}

	let { children, title, icon }: Props = $props();
</script>

<div class="flex-column flex items-center gap-2 border-b pb-2">
	{@render icon({ class: 'h-5 w-5' })}
	<h3 class="text-lg font-semibold">{title}</h3>

	<div class="space-y-6">
		{@render children()}
	</div>
</div>
