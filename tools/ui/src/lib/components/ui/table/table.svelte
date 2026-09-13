<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLTableAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLTableAttributes> = $props();
</script>

<div class="relative w-full overflow-x-auto" data-slot="table-container">
	<table
		bind:this={ref}
		class={cn('w-full caption-bottom text-sm', className)}
		data-slot="table"
		{...restProps}
	>
		{@render children?.()}
	</table>
</div>
