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

<thead
	bind:this={ref}
	class={cn('[&_tr]:border-b', className)}
	data-slot="table-header"
	{...restProps}
>
	{@render children?.()}
</thead>
