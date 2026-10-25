<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div
	bind:this={ref}
	class={cn('col-start-2 row-span-2 row-start-1 self-start justify-self-end', className)}
	data-slot="card-action"
	{...restProps}
>
	{@render children?.()}
</div>
