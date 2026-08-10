<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import { BOX_BORDER } from '$lib/constants';
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
	data-slot="card"
	class={cn(
		'flex flex-col gap-6 rounded-xl bg-card py-6 text-card-foreground shadow-sm',
		BOX_BORDER,
		className
	)}
	{...restProps}
>
	{@render children?.()}
</div>
