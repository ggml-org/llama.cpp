<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
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
	class={cn('flex flex-row gap-2 sm:justify-end [&>*]:flex-1 sm:[&>*]:flex-none', className)}
	data-slot="alert-dialog-footer"
	{...restProps}
>
	{@render children?.()}
</div>
