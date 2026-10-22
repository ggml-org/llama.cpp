<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import type { Snippet } from 'svelte';

	interface Props {
		action?: Snippet;
		children: Snippet;
		title: string;
	}

	let { action, children, title }: Props = $props();
</script>

<div class="flex flex-col gap-4 p-5">
	<div class="flex items-center justify-between gap-2">
		<Card.Title class="text-base">{title}</Card.Title>

		{@render action?.()}
	</div>

	{@render children()}
</div>
