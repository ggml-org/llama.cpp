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
	class={cn('card-surface flex flex-col gap-6 rounded-(--card-radius) py-6', className)}
	data-slot="card"
	{...restProps}
>
	{@render children?.()}
</div>
