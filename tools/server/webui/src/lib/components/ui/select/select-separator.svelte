<script lang="ts">
	import type { Separator as SeparatorPrimitive } from 'bits-ui';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import { cn } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		...restProps
	}: SeparatorPrimitive.RootProps = $props();
</script>

<Separator
	bind:ref
	data-slot="select-separator"
	class={cn('pointer-events-none -mx-1 my-1 h-px bg-border', className)}
	{...restProps}
/>
