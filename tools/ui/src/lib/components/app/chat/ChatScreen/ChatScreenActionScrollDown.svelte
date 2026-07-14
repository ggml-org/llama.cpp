<script lang="ts">
	import { ArrowDown } from '@lucide/svelte';
	import ActionIcon from '$lib/components/app/actions/ActionIcon.svelte';

	let { onclick }: { onclick: (e?: MouseEvent) => void } = $props();
</script>

<div class="pointer-events-auto flex justify-center relative h-0">
	<ActionIcon
		icon={ArrowDown}
		{onclick}
		ariaLabel="Scroll to bottom"
		tooltip="Scroll to bottom"
		size="lg"
		iconSize="h-4 w-4"
		class="h-9 w-9 rounded-full bg-accent text-accent-foreground absolute bottom-4 shadow-md"
	/>
</div>
