<script lang="ts">
	import { Separator } from '$lib/components/ui/separator/index.js';
	import { cn } from '$lib/components/ui/utils.js';
	import type { Separator as SeparatorPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: SeparatorPrimitive.RootProps = $props();
</script>

<Separator
	bind:ref
	class={cn('pointer-events-none -mx-1 my-1 h-px bg-border', className)}
	data-slot="select-separator"
	{...restProps}
/>
