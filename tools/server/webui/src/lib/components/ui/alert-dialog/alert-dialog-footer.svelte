<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div
	bind:this={ref}
	data-slot="alert-dialog-footer"
	class={cn(
		'mt-6 flex flex-row gap-2 sm:mt-0 sm:justify-end [&>*]:flex-1 sm:[&>*]:flex-none',
		className
	)}
	{...restProps}
>
	{@render children?.()}
</div>
