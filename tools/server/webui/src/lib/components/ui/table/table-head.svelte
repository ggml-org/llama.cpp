<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLThAttributes } from 'svelte/elements';

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLThAttributes> = $props();
</script>

<th
	bind:this={ref}
	data-slot="table-head"
	class={cn(
		'h-10 bg-clip-padding px-2 text-left align-middle font-medium whitespace-nowrap text-foreground [&:has([role=checkbox])]:pe-0',
		className
	)}
	{...restProps}
>
	{@render children?.()}
</th>
