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
	data-slot="dialog-header"
	class={cn(
		'flex flex-col gap-2 text-center sm:text-left sticky top-0 z-50 bg-background md:bg-transparent',
		className
	)}
	{...restProps}
>
	{@render children?.()}
</div>
