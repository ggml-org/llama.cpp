<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
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
		iconSize={ICON_CLASS_DEFAULT}
		class="h-9 w-9 rounded-full bg-accent text-accent-foreground absolute bottom-4 shadow-md"
	/>
</div>
