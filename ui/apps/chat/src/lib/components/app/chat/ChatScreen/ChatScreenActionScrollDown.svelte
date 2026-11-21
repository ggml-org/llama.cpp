<script lang="ts">
	import { ArrowDown } from '@lucide/svelte';
	import GlassKey from '#lib/components/app/actions/GlassKey.svelte';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';

	let { onclick }: { onclick: (e?: MouseEvent) => void } = $props();
</script>

<div class="pointer-events-auto flex justify-center relative h-0">
	<GlassKey
		class="absolute bottom-4"
		icon={ArrowDown}
		iconClass={ICON_CLASS_DEFAULT}
		label="Scroll to bottom"
		{onclick}
		size="sm"
		tooltip
	/>
</div>
