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
	class={cn('matte-glass flex flex-col gap-6 rounded-xl py-6 text-card-foreground', className)}
	data-slot="card"
	{...restProps}
>
	{@render children?.()}
</div>
