<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLTableSectionElement>> = $props();
</script>

<tfoot
	bind:this={ref}
	class={cn('border-t bg-muted/50 font-medium [&>tr]:last:border-b-0', className)}
	data-slot="table-footer"
	{...restProps}
>
	{@render children?.()}
</tfoot>
